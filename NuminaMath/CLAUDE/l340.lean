import Mathlib

namespace NUMINAMATH_CALUDE_stamp_coverage_possible_l340_34099

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)

/-- Represents a stamp with black cells -/
structure Stamp (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)
  (black_count : ℕ)
  (black_count_eq : black_count = 102)

/-- Applies a stamp to a grid at a specific position -/
def apply_stamp (g : Grid n) (s : Stamp m) (pos_x pos_y : ℕ) : Grid n :=
  sorry

/-- Checks if a grid is fully covered except for one corner -/
def is_covered_except_corner (g : Grid n) : Prop :=
  sorry

/-- Main theorem: It's possible to cover a 101x101 grid except for one corner
    using a 102-cell stamp 100 times -/
theorem stamp_coverage_possible :
  ∃ (g : Grid 101) (s : Stamp 102) (stamps : List (ℕ × ℕ)),
    stamps.length = 100 ∧
    is_covered_except_corner (stamps.foldl (λ acc (x, y) => apply_stamp acc s x y) g) :=
  sorry

end NUMINAMATH_CALUDE_stamp_coverage_possible_l340_34099


namespace NUMINAMATH_CALUDE_line_through_points_l340_34080

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the line passing through the given points
def line_equation (x : ℝ) : ℝ := 3 * x + 2

-- Define the given points
def p1 : Point := ⟨2, 8⟩
def p2 : Point := ⟨4, 14⟩
def p3 : Point := ⟨6, 20⟩
def p4 : Point := ⟨35, line_equation 35⟩

-- Theorem statement
theorem line_through_points :
  p1.y = line_equation p1.x ∧
  p2.y = line_equation p2.x ∧
  p3.y = line_equation p3.x ∧
  p4.y = 107 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l340_34080


namespace NUMINAMATH_CALUDE_attendees_equal_22_l340_34044

/-- Represents the total number of people who attended a performance given ticket prices and total revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - num_children * child_price) / adult_price
  num_adults + num_children

/-- Theorem stating that given the specific conditions, the total number of attendees is 22 --/
theorem attendees_equal_22 :
  total_attendees 8 1 50 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_attendees_equal_22_l340_34044


namespace NUMINAMATH_CALUDE_problem1_l340_34009

theorem problem1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

end NUMINAMATH_CALUDE_problem1_l340_34009


namespace NUMINAMATH_CALUDE_min_value_theorem_l340_34042

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 2/y + 3/z = 1) :
  x + y/2 + z/3 ≥ 9 ∧ (x + y/2 + z/3 = 9 ↔ x = 3 ∧ y = 6 ∧ z = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l340_34042


namespace NUMINAMATH_CALUDE_tan_neg_585_deg_l340_34035

theorem tan_neg_585_deg : Real.tan (-585 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_neg_585_deg_l340_34035


namespace NUMINAMATH_CALUDE_xy_range_and_min_x_plus_2y_l340_34043

theorem xy_range_and_min_x_plus_2y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x*y = 3) : 
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b = 3 → x + 2*y ≤ a + 2*b) ∧
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + d + c*d = 3 ∧ c + 2*d = 4*Real.sqrt 2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_xy_range_and_min_x_plus_2y_l340_34043


namespace NUMINAMATH_CALUDE_skaters_practice_hours_l340_34025

/-- Represents the practice hours for each skater -/
structure SkaterHours where
  hannah_weekend : ℕ
  hannah_weekday : ℕ
  sarah_weekday : ℕ
  sarah_weekend : ℕ
  emma_weekday : ℕ
  emma_weekend : ℕ

/-- Calculates the total practice hours for all skaters -/
def total_practice_hours (hours : SkaterHours) : ℕ :=
  hours.hannah_weekend + hours.hannah_weekday +
  hours.sarah_weekday + hours.sarah_weekend +
  hours.emma_weekday + hours.emma_weekend

/-- Theorem stating the total practice hours for the skaters -/
theorem skaters_practice_hours :
  ∃ (hours : SkaterHours),
    hours.hannah_weekend = 8 ∧
    hours.hannah_weekday = hours.hannah_weekend + 17 ∧
    hours.sarah_weekday = 12 ∧
    hours.sarah_weekend = 6 ∧
    hours.emma_weekday = 2 * hours.sarah_weekday ∧
    hours.emma_weekend = hours.sarah_weekend + 5 ∧
    total_practice_hours hours = 86 := by
  sorry

end NUMINAMATH_CALUDE_skaters_practice_hours_l340_34025


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l340_34011

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 8 = 0}
def N (p q : ℝ) : Set ℝ := {x | x^2 - q*x + p = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N p q = {1} → p + q = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l340_34011


namespace NUMINAMATH_CALUDE_election_votes_total_l340_34074

theorem election_votes_total (total votes_in_favor votes_against votes_neutral : ℕ) : 
  votes_in_favor = votes_against + 78 →
  votes_against = (375 * total) / 1000 →
  votes_neutral = (125 * total) / 1000 →
  total = votes_in_favor + votes_against + votes_neutral →
  total = 624 := by
sorry

end NUMINAMATH_CALUDE_election_votes_total_l340_34074


namespace NUMINAMATH_CALUDE_modular_arithmetic_properties_l340_34051

theorem modular_arithmetic_properties (a b c d m : ℤ) 
  (h1 : a ≡ b [ZMOD m]) 
  (h2 : c ≡ d [ZMOD m]) : 
  (a + c ≡ b + d [ZMOD m]) ∧ (a * c ≡ b * d [ZMOD m]) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_properties_l340_34051


namespace NUMINAMATH_CALUDE_essay_word_limit_l340_34045

/-- The word limit for Vinnie's essay --/
def word_limit (saturday_words sunday_words exceeded_words : ℕ) : ℕ :=
  saturday_words + sunday_words - exceeded_words

/-- Theorem: The word limit for Vinnie's essay is 1000 words --/
theorem essay_word_limit :
  word_limit 450 650 100 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_limit_l340_34045


namespace NUMINAMATH_CALUDE_sphere_hemisphere_radius_equality_l340_34036

/-- The radius of a sphere is equal to the radius of each of two hemispheres 
    that have the same total volume as the original sphere. -/
theorem sphere_hemisphere_radius_equality (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3) = (2 * (2 / 3 * Real.pi * r^3)) := by
  sorry

#check sphere_hemisphere_radius_equality

end NUMINAMATH_CALUDE_sphere_hemisphere_radius_equality_l340_34036


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l340_34094

theorem lcm_factor_problem (A B : ℕ) (X : ℕ+) :
  A = 400 →
  Nat.gcd A B = 25 →
  Nat.lcm A B = 25 * X * 16 →
  X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l340_34094


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l340_34064

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Fin 10

/-- The probability of choosing 3 distinct vertices from a decagon that form a triangle
    with sides that are all edges of the decagon -/
theorem decagon_triangle_probability : 
  (Nat.choose 10 3 : ℚ)⁻¹ * 10 = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l340_34064


namespace NUMINAMATH_CALUDE_curve_crosses_itself_l340_34017

/-- The x-coordinate of a point on the curve -/
def x (t k : ℝ) : ℝ := t^2 + k

/-- The y-coordinate of a point on the curve -/
def y (t k : ℝ) : ℝ := t^3 - k*t + 5

/-- Theorem stating that the curve crosses itself at (18,5) when k = 9 -/
theorem curve_crosses_itself : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ 9 = x t₂ 9 ∧ 
    y t₁ 9 = y t₂ 9 ∧
    x t₁ 9 = 18 ∧ 
    y t₁ 9 = 5 :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_itself_l340_34017


namespace NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l340_34033

theorem inequality_upper_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 4 / Real.sqrt 3 :=
by sorry

theorem upper_bound_tight : 
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) = 4 / Real.sqrt 3 :=
by sorry

theorem smallest_upper_bound :
  ∀ M : ℝ, M < 4 / Real.sqrt 3 →
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > M :=
by sorry

end NUMINAMATH_CALUDE_inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l340_34033


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l340_34005

/-- Proves that given a mixture with 10% water content, if 5 liters of water are added
    to make the new mixture contain 20% water, then the initial volume of the mixture was 40 liters. -/
theorem initial_mixture_volume
  (initial_water_percentage : Real)
  (added_water : Real)
  (final_water_percentage : Real)
  (h1 : initial_water_percentage = 0.10)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 0.20)
  : ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water
      = (initial_volume + added_water) * final_water_percentage
    ∧ initial_volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l340_34005


namespace NUMINAMATH_CALUDE_fountain_area_l340_34083

theorem fountain_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : ∃ (r : ℝ), r^2 = 244 ∧ π * r^2 = 244 * π := by
  sorry

end NUMINAMATH_CALUDE_fountain_area_l340_34083


namespace NUMINAMATH_CALUDE_jackson_holidays_l340_34061

/-- The number of holidays Jackson takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The total number of holidays Jackson takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_per_year

theorem jackson_holidays : total_holidays = 36 := by
  sorry

end NUMINAMATH_CALUDE_jackson_holidays_l340_34061


namespace NUMINAMATH_CALUDE_sam_spent_three_dimes_per_candy_bar_l340_34012

/-- Represents the number of cents in a dime -/
def dime_value : ℕ := 10

/-- Represents the number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- Represents the initial number of dimes Sam has -/
def initial_dimes : ℕ := 19

/-- Represents the initial number of quarters Sam has -/
def initial_quarters : ℕ := 6

/-- Represents the number of candy bars Sam buys -/
def candy_bars : ℕ := 4

/-- Represents the number of lollipops Sam buys -/
def lollipops : ℕ := 1

/-- Represents the amount of money Sam has left after purchases (in cents) -/
def money_left : ℕ := 195

/-- Proves that Sam spent 3 dimes on each candy bar -/
theorem sam_spent_three_dimes_per_candy_bar :
  ∃ (dimes_per_candy : ℕ),
    dimes_per_candy * candy_bars * dime_value + 
    lollipops * quarter_value + 
    money_left = 
    initial_dimes * dime_value + 
    initial_quarters * quarter_value ∧
    dimes_per_candy = 3 := by
  sorry

end NUMINAMATH_CALUDE_sam_spent_three_dimes_per_candy_bar_l340_34012


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_of_exponents_l340_34006

/-- Two terms are considered like terms if their variables and corresponding exponents match -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (a b : ℕ), ∃ (c : ℚ), term1 a b = c * term2 a b ∨ term2 a b = c * term1 a b

/-- The first term in our problem -/
def term1 (m : ℕ) (a b : ℕ) : ℚ := 2 * (a ^ m) * (b ^ 3)

/-- The second term in our problem -/
def term2 (n : ℕ) (a b : ℕ) : ℚ := -3 * a * (b ^ n)

theorem like_terms_imply_sum_of_exponents (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m + n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_of_exponents_l340_34006


namespace NUMINAMATH_CALUDE_cost_to_fly_D_to_E_l340_34007

/-- Represents a city in the triangle --/
inductive City
| D
| E
| F

/-- Calculates the cost of flying between two cities --/
def flyCost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The triangle formed by the cities --/
structure Triangle where
  DE : ℝ
  DF : ℝ
  isRightAngled : True

/-- The problem setup --/
structure TripProblem where
  cities : Triangle
  flyFromDToE : True

theorem cost_to_fly_D_to_E (problem : TripProblem) : 
  flyCost problem.cities.DE = 660 :=
sorry

end NUMINAMATH_CALUDE_cost_to_fly_D_to_E_l340_34007


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l340_34092

theorem parallel_vectors_y_value :
  ∀ (y : ℝ),
  let a : Fin 2 → ℝ := ![(-1), 3]
  let b : Fin 2 → ℝ := ![2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) →
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l340_34092


namespace NUMINAMATH_CALUDE_correct_equation_l340_34063

/-- Represents the meeting problem of two people walking towards each other -/
def meeting_problem (total_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  time * (speed1 + speed2) = total_distance

theorem correct_equation : 
  let total_distance : ℝ := 25
  let time : ℝ := 3
  let speed1 : ℝ := 4
  let speed2 : ℝ := x
  meeting_problem total_distance time speed1 speed2 ↔ 3 * (4 + x) = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l340_34063


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l340_34040

/-- Given three points that represent three of the four endpoints of the axes of an ellipse -/
def point1 : ℝ × ℝ := (-2, 4)
def point2 : ℝ × ℝ := (3, -2)
def point3 : ℝ × ℝ := (8, 4)

/-- The theorem stating that the distance between the foci of the ellipse is 2√11 -/
theorem ellipse_foci_distance :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (center.1 - a = point1.1 ∨ center.1 - a = point2.1 ∨ center.1 - a = point3.1) ∧
    (center.1 + a = point1.1 ∨ center.1 + a = point2.1 ∨ center.1 + a = point3.1) ∧
    (center.2 - b = point1.2 ∨ center.2 - b = point2.2 ∨ center.2 - b = point3.2) ∧
    (center.2 + b = point1.2 ∨ center.2 + b = point2.2 ∨ center.2 + b = point3.2) ∧
    2 * Real.sqrt (max a b ^ 2 - min a b ^ 2) = 2 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l340_34040


namespace NUMINAMATH_CALUDE_popped_kernel_red_probability_l340_34090

theorem popped_kernel_red_probability
  (total_kernels : ℝ)
  (red_ratio : ℝ)
  (green_ratio : ℝ)
  (red_pop_ratio : ℝ)
  (green_pop_ratio : ℝ)
  (h1 : red_ratio = 3/4)
  (h2 : green_ratio = 1/4)
  (h3 : red_pop_ratio = 3/5)
  (h4 : green_pop_ratio = 3/4)
  (h5 : red_ratio + green_ratio = 1) :
  let red_kernels := red_ratio * total_kernels
  let green_kernels := green_ratio * total_kernels
  let popped_red := red_pop_ratio * red_kernels
  let popped_green := green_pop_ratio * green_kernels
  let total_popped := popped_red + popped_green
  (popped_red / total_popped) = 12/17 :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_red_probability_l340_34090


namespace NUMINAMATH_CALUDE_flight_distance_difference_l340_34024

def beka_flights : List Nat := [425, 320, 387]
def jackson_flights : List Nat := [250, 170, 353, 201]

theorem flight_distance_difference :
  (List.sum beka_flights) - (List.sum jackson_flights) = 158 := by
  sorry

end NUMINAMATH_CALUDE_flight_distance_difference_l340_34024


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l340_34039

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l340_34039


namespace NUMINAMATH_CALUDE_integral_equality_l340_34047

theorem integral_equality : ∫ x in (1 : ℝ)..Real.sqrt 3, 
  (x^(2*x^2 + 1) + Real.log (x^(2*x^(2*x^2 + 1)))) = 13 := by sorry

end NUMINAMATH_CALUDE_integral_equality_l340_34047


namespace NUMINAMATH_CALUDE_player_B_wins_l340_34030

/-- Represents the state of the pizza game -/
structure GameState :=
  (pizzeria1 : Nat)
  (pizzeria2 : Nat)

/-- Represents a player's move -/
inductive Move
  | EatFromOne (pizzeria : Nat) (amount : Nat)
  | EatFromBoth

/-- Defines the rules of the game -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.EatFromOne 1 amount => amount > 0 ∧ amount ≤ state.pizzeria1
  | Move.EatFromOne 2 amount => amount > 0 ∧ amount ≤ state.pizzeria2
  | Move.EatFromBoth => state.pizzeria1 > 0 ∧ state.pizzeria2 > 0
  | _ => False

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.EatFromOne 1 amount => ⟨state.pizzeria1 - amount, state.pizzeria2⟩
  | Move.EatFromOne 2 amount => ⟨state.pizzeria1, state.pizzeria2 - amount⟩
  | Move.EatFromBoth => ⟨state.pizzeria1 - 1, state.pizzeria2 - 1⟩
  | _ => state

/-- Defines a winning strategy for player B -/
def hasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    (state.pizzeria1 = 2010 ∧ state.pizzeria2 = 2010) →
    ∃ (strategy : GameState → Move),
      (∀ (s : GameState), isValidMove s (strategy s)) ∧
      (player = 2 → ∃ (n : Nat), state.pizzeria1 + state.pizzeria2 = n ∧ n % 2 = 1)

/-- The main theorem: Player B (second player) has a winning strategy -/
theorem player_B_wins : hasWinningStrategy 2 := by
  sorry

end NUMINAMATH_CALUDE_player_B_wins_l340_34030


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l340_34027

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (h : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by
  sorry

#check oblique_triangular_prism_volume

end NUMINAMATH_CALUDE_oblique_triangular_prism_volume_l340_34027


namespace NUMINAMATH_CALUDE_complex_product_real_iff_condition_l340_34075

theorem complex_product_real_iff_condition (a b c d : ℝ) :
  let Z1 : ℂ := Complex.mk a b
  let Z2 : ℂ := Complex.mk c d
  (Z1 * Z2).im = 0 ↔ a * d + b * c = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_iff_condition_l340_34075


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_is_117_143_l340_34023

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zincCopperRatio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def givenMixture : Mixture :=
  { totalWeight := 78
  , zincWeight := 35.1 }

/-- Theorem stating that the ratio of zinc to copper in the given mixture is 117:143 -/
theorem zinc_copper_ratio_is_117_143 :
  zincCopperRatio givenMixture = { numerator := 117, denominator := 143 } :=
  sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_is_117_143_l340_34023


namespace NUMINAMATH_CALUDE_arrange_85550_eq_16_l340_34038

/-- The number of ways to arrange the digits of 85550 to form a 5-digit number -/
def arrange_85550 : ℕ :=
  let digits : Multiset ℕ := {8, 5, 5, 5, 0}
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  16

/-- Theorem stating that the number of ways to arrange the digits of 85550
    to form a 5-digit number is 16 -/
theorem arrange_85550_eq_16 : arrange_85550 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arrange_85550_eq_16_l340_34038


namespace NUMINAMATH_CALUDE_function_derivative_positive_l340_34028

/-- Given a function y = 2mx^2 + (1-4m)x + 2m - 1, prove that when m = -1 and x < 5/4, 
    the derivative of y with respect to x is positive. -/
theorem function_derivative_positive (x : ℝ) (h : x < 5/4) : 
  let m : ℝ := -1
  let y : ℝ → ℝ := λ x => 2*m*x^2 + (1-4*m)*x + 2*m - 1
  (deriv y) x > 0 := by sorry

end NUMINAMATH_CALUDE_function_derivative_positive_l340_34028


namespace NUMINAMATH_CALUDE_fixed_distance_point_l340_34067

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given vectors a and b, and a vector p satisfying ||p - b|| = 3||p - a||,
    p is at a fixed distance from (9/8)a + (-1/8)b. -/
theorem fixed_distance_point (a b p : V) 
    (h : ‖p - b‖ = 3 * ‖p - a‖) : 
    ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a + (-1/8 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_fixed_distance_point_l340_34067


namespace NUMINAMATH_CALUDE_math_test_score_l340_34048

theorem math_test_score (korean_score english_score : ℕ)
  (h1 : (korean_score + english_score) / 2 = 92)
  (h2 : (korean_score + english_score + math_score) / 3 = 94)
  : math_score = 98 := by
  sorry

end NUMINAMATH_CALUDE_math_test_score_l340_34048


namespace NUMINAMATH_CALUDE_clock_angle_at_6_30_l340_34003

/-- The smaller angle between the hour and minute hands of a clock at 6:30 -/
def clock_angle : ℝ :=
  let hour_hand_rate : ℝ := 0.5  -- degrees per minute
  let minute_hand_rate : ℝ := 6  -- degrees per minute
  let time_passed : ℝ := 30      -- minutes since 6:00
  let hour_hand_position : ℝ := hour_hand_rate * time_passed
  let minute_hand_position : ℝ := minute_hand_rate * time_passed
  minute_hand_position - hour_hand_position

theorem clock_angle_at_6_30 : clock_angle = 15 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_6_30_l340_34003


namespace NUMINAMATH_CALUDE_president_and_committee_from_ten_l340_34052

/-- The number of ways to choose a president and a committee from a group --/
def choose_president_and_committee (group_size : ℕ) (committee_size : ℕ) : ℕ :=
  group_size * Nat.choose (group_size - 1) committee_size

/-- Theorem stating the number of ways to choose a president and a 3-person committee from 10 people --/
theorem president_and_committee_from_ten :
  choose_president_and_committee 10 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_from_ten_l340_34052


namespace NUMINAMATH_CALUDE_equal_spacing_ratio_l340_34085

/-- Given 6 equally spaced points on a number line from 0 to 1, 
    the ratio of the 3rd point's value to the 6th point's value is 0.5 -/
theorem equal_spacing_ratio : 
  ∀ (P Q R S T U : ℝ), 
    0 ≤ P ∧ P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U = 1 →
    Q - P = R - Q ∧ R - Q = S - R ∧ S - R = T - S ∧ T - S = U - T →
    R / U = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_spacing_ratio_l340_34085


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l340_34088

def I : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {3,4,5}
def B : Set Nat := {1,3,6}

theorem complement_intersection_theorem : 
  (I \ A) ∩ (I \ B) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l340_34088


namespace NUMINAMATH_CALUDE_no_real_roots_l340_34013

theorem no_real_roots (a : ℝ) : (∀ x : ℝ, |x| ≠ a * x + 1) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l340_34013


namespace NUMINAMATH_CALUDE_randy_quiz_average_l340_34056

/-- The number of quizzes Randy wants to have the average for -/
def n : ℕ := 5

/-- The sum of Randy's first four quiz scores -/
def initial_sum : ℕ := 374

/-- Randy's desired average -/
def desired_average : ℕ := 94

/-- Randy's next quiz score -/
def next_score : ℕ := 96

theorem randy_quiz_average : 
  (initial_sum + next_score : ℚ) / n = desired_average := by sorry

end NUMINAMATH_CALUDE_randy_quiz_average_l340_34056


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l340_34026

theorem simplify_trig_expression : 
  Real.sqrt (1 + Real.sin 10) + Real.sqrt (1 - Real.sin 10) = -2 * Real.sin 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l340_34026


namespace NUMINAMATH_CALUDE_counterexample_exists_l340_34055

theorem counterexample_exists : ∃ x y : ℝ, x + y = 5 ∧ ¬(x = 1 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l340_34055


namespace NUMINAMATH_CALUDE_problem_solution_l340_34020

theorem problem_solution (a b c : ℝ) 
  (sum_eq : a + b + c = 99)
  (equal_after_change : a + 6 = b - 6 ∧ b - 6 = 5 * c) : 
  b = 51 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l340_34020


namespace NUMINAMATH_CALUDE_Z_equals_S_l340_34089

-- Define the set of functions F
def F : Set (ℝ → ℝ) := {f | ∀ x y, f (x + f y) = f x + f y}

-- Define the set of rational numbers q
def Z : Set ℚ := {q | ∀ f ∈ F, ∃ z : ℝ, f z = q * z}

-- Define the set S
def S : Set ℚ := {q | ∃ n : ℤ, n ≠ 0 ∧ q = (n + 1) / n}

-- State the theorem
theorem Z_equals_S : Z = S := by sorry

end NUMINAMATH_CALUDE_Z_equals_S_l340_34089


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l340_34081

theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  (a:ℝ) + (b:ℝ) = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l340_34081


namespace NUMINAMATH_CALUDE_circle_within_circle_l340_34066

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle is contained within another circle if all its points are inside the larger circle -/
def is_contained (inner outer : Circle) : Prop :=
  ∀ p : ℝ × ℝ, is_inside p inner → is_inside p outer

theorem circle_within_circle (C : Circle) (A B : ℝ × ℝ) 
    (hA : is_inside A C) (hB : is_inside B C) :
  ∃ D : Circle, is_inside A D ∧ is_inside B D ∧ is_contained D C := by
  sorry

end NUMINAMATH_CALUDE_circle_within_circle_l340_34066


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l340_34018

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = x^2 + ax + b -/
def g (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The condition of symmetry about x = 1 -/
def symmetry_condition (a b : ℝ) : Prop :=
  ∀ x, g a b x = f (2 - x)

/-- Theorem: If f and g are symmetrical about x = 1, then a = -4 and b = 4 -/
theorem symmetry_implies_values :
  ∀ a b, symmetry_condition a b → a = -4 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l340_34018


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l340_34002

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + 4/b < 6 ∧ b + 9/c < 6 ∧ c + 16/a < 6) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l340_34002


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l340_34073

/-- An isosceles triangle with specific heights -/
structure IsoscelesTriangle where
  -- The height drawn to the base
  baseHeight : ℝ
  -- The height drawn to one of the equal sides
  sideHeight : ℝ
  -- Assumption that the triangle is isosceles
  isIsosceles : True

/-- The base of the triangle -/
def baseLength (triangle : IsoscelesTriangle) : ℝ :=
  7.5

/-- Theorem stating that for an isosceles triangle with given heights, the base length is 7.5 -/
theorem isosceles_triangle_base_length 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.baseHeight = 5) 
  (h2 : triangle.sideHeight = 6) : 
  baseLength triangle = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l340_34073


namespace NUMINAMATH_CALUDE_uncles_age_l340_34034

theorem uncles_age (bud_age uncle_age : ℕ) : 
  bud_age = 8 → 
  3 * bud_age = uncle_age → 
  uncle_age = 24 := by
sorry

end NUMINAMATH_CALUDE_uncles_age_l340_34034


namespace NUMINAMATH_CALUDE_other_frisbee_price_is_3_l340_34082

/-- Represents the price and sales of frisbees in a sporting goods store --/
structure FrisbeeSales where
  total_sold : ℕ
  total_receipts : ℕ
  price_other : ℚ
  min_sold_at_4 : ℕ

/-- Checks if the given FrisbeeSales satisfies the problem conditions --/
def is_valid_sale (sale : FrisbeeSales) : Prop :=
  sale.total_sold = 60 ∧
  sale.total_receipts = 204 ∧
  sale.min_sold_at_4 = 24 ∧
  sale.price_other * (sale.total_sold - sale.min_sold_at_4) + 4 * sale.min_sold_at_4 = sale.total_receipts

/-- Theorem stating that the price of the other frisbees is $3 --/
theorem other_frisbee_price_is_3 :
  ∀ (sale : FrisbeeSales), is_valid_sale sale → sale.price_other = 3 := by
  sorry

end NUMINAMATH_CALUDE_other_frisbee_price_is_3_l340_34082


namespace NUMINAMATH_CALUDE_one_root_quadratic_l340_34019

theorem one_root_quadratic (k : ℝ) : 
  (∃! x : ℝ, k * x^2 - 8 * x + 16 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_root_quadratic_l340_34019


namespace NUMINAMATH_CALUDE_abc_reciprocal_sum_l340_34096

theorem abc_reciprocal_sum (a b c : ℝ) 
  (h1 : a + 1/b = 9)
  (h2 : b + 1/c = 10)
  (h3 : c + 1/a = 11) :
  a * b * c + 1 / (a * b * c) = 960 := by
  sorry

end NUMINAMATH_CALUDE_abc_reciprocal_sum_l340_34096


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l340_34071

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x ≥ 2 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x = 2 * Real.sqrt 3 ↔ 
  (x = Real.sqrt (Real.sqrt 3) ∨ x = -Real.sqrt (Real.sqrt 3)) ∧
  (y = -1 / x) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l340_34071


namespace NUMINAMATH_CALUDE_locus_of_tangent_points_l340_34086

/-- Given a parabola y^2 = 2px and a constant k, prove that the locus of points P(x, y) 
    from which tangents can be drawn to the parabola with slopes m1 and m2 satisfying 
    m1 * m2^2 + m1^2 * m2 = k, is the parabola x^2 = (p / (2k)) * y -/
theorem locus_of_tangent_points (p k : ℝ) (hp : p > 0) (hk : k ≠ 0) :
  ∀ x y m1 m2 : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    y1^2 = 2 * p * x1 ∧ 
    y2^2 = 2 * p * x2 ∧
    m1 = p / y1 ∧
    m2 = p / y2 ∧
    2 * y = y1 + y2 ∧
    x^2 = x1 * x2 ∧
    m1 * m2^2 + m1^2 * m2 = k) →
  x^2 = (p / (2 * k)) * y := by
  sorry

end NUMINAMATH_CALUDE_locus_of_tangent_points_l340_34086


namespace NUMINAMATH_CALUDE_inequality_proof_l340_34041

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hpqr : p * q * r = 1) :
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l340_34041


namespace NUMINAMATH_CALUDE_three_leaf_clover_count_l340_34093

/-- The number of leaves on a three-leaf clover -/
def three_leaf_count : ℕ := 3

/-- The number of leaves on a four-leaf clover -/
def four_leaf_count : ℕ := 4

/-- The total number of leaves collected -/
def total_leaves : ℕ := 100

/-- The number of four-leaf clovers found -/
def four_leaf_clovers : ℕ := 1

theorem three_leaf_clover_count :
  (total_leaves - four_leaf_count * four_leaf_clovers) / three_leaf_count = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_leaf_clover_count_l340_34093


namespace NUMINAMATH_CALUDE_inequality_proof_l340_34076

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 3) :
  1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l340_34076


namespace NUMINAMATH_CALUDE_composite_surface_area_is_39_l340_34022

/-- The surface area of a composite object formed by three cylinders -/
def composite_surface_area (π : ℝ) (h : ℝ) (r₁ r₂ r₃ : ℝ) : ℝ :=
  (2 * π * r₁ * h + π * r₁^2) +
  (2 * π * r₂ * h + π * r₂^2) +
  (2 * π * r₃ * h + π * r₃^2) +
  π * r₁^2 + π * r₂^2 + π * r₃^2

/-- The surface area of the composite object is 39 square meters -/
theorem composite_surface_area_is_39 :
  composite_surface_area 3 1 1.5 1 0.5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_composite_surface_area_is_39_l340_34022


namespace NUMINAMATH_CALUDE_smallest_value_u3_plus_v3_l340_34010

theorem smallest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 11) : 
  Complex.abs (u^3 + v^3) ≥ 14.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_u3_plus_v3_l340_34010


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l340_34032

theorem solve_exponential_equation :
  ∃! x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) :=
by
  use -10
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l340_34032


namespace NUMINAMATH_CALUDE_functional_equation_solution_l340_34070

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  g 0 = 1 ∧ ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y

/-- The main theorem stating that g(x) = 5^x - 3^x is the unique solution -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∀ x : ℝ, g x = 5^x - 3^x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l340_34070


namespace NUMINAMATH_CALUDE_age_difference_l340_34000

/-- Proves that Sachin is 8 years younger than Rahul given their age ratio and Sachin's age -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 28 →
  (sachin_age : ℚ) / rahul_age = 7 / 9 →
  rahul_age - sachin_age = 8 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l340_34000


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l340_34084

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l340_34084


namespace NUMINAMATH_CALUDE_inequality_proof_l340_34098

theorem inequality_proof (x y z : ℝ) 
  (h1 : -2 ≤ x ∧ x ≤ 2) 
  (h2 : -2 ≤ y ∧ y ≤ 2) 
  (h3 : -2 ≤ z ∧ z ≤ 2) 
  (h4 : x^2 + y^2 + z^2 + x*y*z = 4) : 
  z * (x*z + y*z + y) / (x*y + y^2 + z^2 + 1) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l340_34098


namespace NUMINAMATH_CALUDE_kids_difference_l340_34031

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 6) 
  (h2 : tuesday = 5) : 
  monday - tuesday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l340_34031


namespace NUMINAMATH_CALUDE_tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l340_34049

theorem tan_45_plus_half_inv_plus_abs_neg_two_equals_five :
  Real.tan (π / 4) + (1 / 2)⁻¹ + |(-2)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l340_34049


namespace NUMINAMATH_CALUDE_vector_magnitude_l340_34029

def a : ℝ × ℝ := (-2, -1)

theorem vector_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10) 
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l340_34029


namespace NUMINAMATH_CALUDE_first_house_price_correct_l340_34015

/-- Represents the price of Tommy's first house in dollars -/
def first_house_price : ℝ := 400000

/-- Represents the price of Tommy's new house in dollars -/
def new_house_price : ℝ := 500000

/-- Represents the loan percentage for the new house -/
def loan_percentage : ℝ := 0.75

/-- Represents the annual interest rate for the loan -/
def annual_interest_rate : ℝ := 0.035

/-- Represents the loan term in years -/
def loan_term : ℕ := 15

/-- Represents the annual property tax rate -/
def property_tax_rate : ℝ := 0.015

/-- Represents the annual home insurance cost in dollars -/
def annual_insurance_cost : ℝ := 7500

/-- Theorem stating that the first house price is correct given the conditions -/
theorem first_house_price_correct :
  first_house_price = new_house_price / 1.25 ∧
  new_house_price = first_house_price * 1.25 ∧
  loan_percentage * new_house_price * annual_interest_rate +
  property_tax_rate * new_house_price +
  annual_insurance_cost =
  28125 :=
sorry

end NUMINAMATH_CALUDE_first_house_price_correct_l340_34015


namespace NUMINAMATH_CALUDE_light_bulb_probabilities_l340_34079

/-- Market share of Factory A -/
def market_share_A : ℝ := 0.6

/-- Market share of Factory B -/
def market_share_B : ℝ := 0.4

/-- Qualification rate of Factory A products -/
def qual_rate_A : ℝ := 0.9

/-- Qualification rate of Factory B products -/
def qual_rate_B : ℝ := 0.8

/-- Probability of exactly one qualified light bulb out of two from Factory A -/
def prob_one_qualified_A : ℝ := 2 * qual_rate_A * (1 - qual_rate_A)

/-- Probability of a randomly purchased light bulb being qualified -/
def prob_random_qualified : ℝ := market_share_A * qual_rate_A + market_share_B * qual_rate_B

theorem light_bulb_probabilities :
  prob_one_qualified_A = 0.18 ∧ prob_random_qualified = 0.86 := by
  sorry

#check light_bulb_probabilities

end NUMINAMATH_CALUDE_light_bulb_probabilities_l340_34079


namespace NUMINAMATH_CALUDE_line_intersects_circle_l340_34072

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length when the line intersects the circle -/
def chord_length : ℝ := 4

/-- The theorem statement -/
theorem line_intersects_circle (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y ∧ line_equation a x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ line_equation a x₁ y₁ ∧
    circle_equation x₂ y₂ ∧ line_equation a x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l340_34072


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l340_34062

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) :
  ∃ k : Nat, 1001 * k = a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c :=
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l340_34062


namespace NUMINAMATH_CALUDE_fourDigitPermutationsFromSixIs360_l340_34087

/-- The number of permutations of 4 digits chosen from a set of 6 digits -/
def fourDigitPermutationsFromSix : ℕ :=
  6 * 5 * 4 * 3

/-- Theorem stating that the number of four-digit numbers without repeating digits
    from the set {1, 2, 3, 4, 5, 6} is equal to 360 -/
theorem fourDigitPermutationsFromSixIs360 : fourDigitPermutationsFromSix = 360 := by
  sorry


end NUMINAMATH_CALUDE_fourDigitPermutationsFromSixIs360_l340_34087


namespace NUMINAMATH_CALUDE_local_extremum_values_l340_34046

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

-- State the theorem
theorem local_extremum_values (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →
  a = -4 ∧ b = 11 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_values_l340_34046


namespace NUMINAMATH_CALUDE_james_letter_frequency_l340_34065

/-- Calculates how many times per week James writes letters to his friends -/
def letters_per_week (pages_per_year : ℕ) (weeks_per_year : ℕ) (pages_per_letter : ℕ) (num_friends : ℕ) : ℕ :=
  (pages_per_year / weeks_per_year) / (pages_per_letter * num_friends)

/-- Theorem stating that James writes letters 2 times per week -/
theorem james_letter_frequency :
  letters_per_week 624 52 3 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_letter_frequency_l340_34065


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l340_34050

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats :
  initial_siamese_cats = 13 ∧
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l340_34050


namespace NUMINAMATH_CALUDE_a_a_a_zero_l340_34069

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_a_a_zero : a (a (a 0)) = 343 := by sorry

end NUMINAMATH_CALUDE_a_a_a_zero_l340_34069


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l340_34053

/-- The interest rate (as a percentage) at which A lent money to B -/
def interest_rate_A_to_B : ℝ := 10

/-- The principal amount lent -/
def principal : ℝ := 3500

/-- The interest rate (as a percentage) at which B lent money to C -/
def interest_rate_B_to_C : ℝ := 15

/-- The time period in years -/
def time : ℝ := 3

/-- B's gain over the time period -/
def B_gain : ℝ := 525

theorem interest_rate_calculation :
  let interest_C := principal * interest_rate_B_to_C / 100 * time
  let interest_A := interest_C - B_gain
  interest_A = principal * interest_rate_A_to_B / 100 * time := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l340_34053


namespace NUMINAMATH_CALUDE_number_of_possible_sets_l340_34001

theorem number_of_possible_sets (A : Set ℤ) : 
  (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (S : Finset (Set ℤ)), (∀ X ∈ S, X ∪ {-1, 1} = {-1, 0, 1}) ∧ S.card = 4 ∧ 
    ∀ Y, Y ∪ {-1, 1} = {-1, 0, 1} → Y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sets_l340_34001


namespace NUMINAMATH_CALUDE_remainder_777_444_mod_11_l340_34008

theorem remainder_777_444_mod_11 : 777^444 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_444_mod_11_l340_34008


namespace NUMINAMATH_CALUDE_least_m_for_x_bound_l340_34037

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_for_x_bound :
  ∃ m : ℕ, m = 33 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k < m, x k > 3 + 1 / 2^10 :=
sorry

end NUMINAMATH_CALUDE_least_m_for_x_bound_l340_34037


namespace NUMINAMATH_CALUDE_a_range_l340_34058

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + 1 = 0}

-- Define the set B
def B : Set ℝ := {1, 2}

-- Define the theorem
theorem a_range (a : ℝ) : (A a ⊆ B) ↔ a ∈ Set.Icc (-2) 2 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l340_34058


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l340_34097

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 1| = |x - 3| :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l340_34097


namespace NUMINAMATH_CALUDE_harry_iguanas_l340_34095

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- The cost to feed each snake per month -/
def snake_feed_cost : ℕ := 10

/-- The cost to feed each iguana per month -/
def iguana_feed_cost : ℕ := 5

/-- The cost to feed each gecko per month -/
def gecko_feed_cost : ℕ := 15

/-- The total yearly cost to feed all pets -/
def yearly_feed_cost : ℕ := 1140

theorem harry_iguanas :
  num_iguanas * iguana_feed_cost * 12 +
  num_geckos * gecko_feed_cost * 12 +
  num_snakes * snake_feed_cost * 12 = yearly_feed_cost :=
by sorry

end NUMINAMATH_CALUDE_harry_iguanas_l340_34095


namespace NUMINAMATH_CALUDE_intersection_line_equation_l340_34077

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B →
    ∀ (P : ℝ × ℝ),
      (∃ t : ℝ, P = t • A + (1 - t) • B) ↔ line P.1 P.2 :=
by sorry


end NUMINAMATH_CALUDE_intersection_line_equation_l340_34077


namespace NUMINAMATH_CALUDE_derivative_of_y_l340_34060

noncomputable def y (x : ℝ) : ℝ := Real.exp (-5 * x + 2)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -5 * Real.exp (-5 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_y_l340_34060


namespace NUMINAMATH_CALUDE_largest_package_size_l340_34021

theorem largest_package_size (a b c : ℕ) (ha : a = 30) (hb : b = 45) (hc : c = 75) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l340_34021


namespace NUMINAMATH_CALUDE_white_balls_count_l340_34054

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l340_34054


namespace NUMINAMATH_CALUDE_cassy_jars_left_l340_34014

/-- The number of jars left unpacked when Cassy fills all boxes -/
def jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
                       (jars_per_box2 : ℕ) (num_boxes2 : ℕ) 
                       (total_jars : ℕ) : ℕ :=
  total_jars - (jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2)

theorem cassy_jars_left :
  jars_left_unpacked 12 10 10 30 500 = 80 := by
  sorry

end NUMINAMATH_CALUDE_cassy_jars_left_l340_34014


namespace NUMINAMATH_CALUDE_max_shape_pairs_l340_34057

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a pair of shapes: a corner and a 2x2 square -/
structure ShapePair where
  corner : Unit
  square : Unit

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area occupied by a single ShapePair -/
def ShapePair.area : ℕ := 7  -- 3 for corner + 4 for 2x2 square

/-- The main theorem to prove -/
theorem max_shape_pairs (r : Rectangle) (h1 : r.width = 3) (h2 : r.height = 100) :
  ∃ (n : ℕ), n = 33 ∧ 
  n * ShapePair.area ≤ r.area ∧
  ∀ (m : ℕ), m * ShapePair.area ≤ r.area → m ≤ n :=
by sorry


end NUMINAMATH_CALUDE_max_shape_pairs_l340_34057


namespace NUMINAMATH_CALUDE_call_duration_is_60_minutes_l340_34016

/-- Represents the duration of a single customer call in minutes. -/
def call_duration (cost_per_minute : ℚ) (monthly_bill : ℚ) (customers_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  (monthly_bill / cost_per_minute) / (customers_per_week * weeks_per_month)

/-- Theorem stating that under the given conditions, each call lasts 60 minutes. -/
theorem call_duration_is_60_minutes :
  call_duration (5 / 100) 600 50 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_call_duration_is_60_minutes_l340_34016


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l340_34059

/-- The line equation x cos θ + y sin θ = 1 is tangent to the circle x² + y² = 1 -/
theorem line_tangent_to_circle :
  ∀ θ : ℝ, 
  (∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 → x^2 + y^2 = 1) ∧
  (∃ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 ∧ x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l340_34059


namespace NUMINAMATH_CALUDE_angle_sum_from_tangents_l340_34078

theorem angle_sum_from_tangents (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 2 → 
  Real.tan β = 3 → 
  α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_from_tangents_l340_34078


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l340_34004

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l340_34004


namespace NUMINAMATH_CALUDE_max_red_socks_l340_34091

/-- The maximum number of red socks in a dresser with specific conditions -/
theorem max_red_socks (t : ℕ) (h1 : t ≤ 2500) :
  let p := 12 / 23
  ∃ r : ℕ, r ≤ t ∧
    (r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = p ∧
    (∀ r' : ℕ, r' ≤ t →
      (r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = p →
      r' ≤ r) ∧
    r = 1225 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l340_34091


namespace NUMINAMATH_CALUDE_office_employees_l340_34068

theorem office_employees (total_employees : ℕ) : 
  (total_employees : ℝ) * 0.25 * 0.6 = 120 → total_employees = 800 := by
  sorry

end NUMINAMATH_CALUDE_office_employees_l340_34068
