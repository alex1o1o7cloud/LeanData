import Mathlib

namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2239_223988

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) 
  (h1 : total = 250)
  (h2 : football = 160)
  (h3 : cricket = 90)
  (h4 : neither = 50) :
  football + cricket - (total - neither) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2239_223988


namespace NUMINAMATH_CALUDE_no_valid_tiling_l2239_223974

/-- Represents a chessboard with one corner removed -/
def ChessboardWithCornerRemoved := Fin 8 × Fin 8

/-- Represents a trimino (3x1 rectangle) -/
def Trimino := Fin 3 × Fin 1

/-- A tiling of the chessboard with triminos -/
def Tiling := ChessboardWithCornerRemoved → Option Trimino

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (t : Tiling) : Prop :=
  -- Each square is either covered by a trimino or is the removed corner
  ∀ (x : ChessboardWithCornerRemoved), 
    (x ≠ (7, 7) → t x ≠ none) ∧ 
    (x = (7, 7) → t x = none) ∧
  -- Each trimino covers exactly three squares
  ∀ (p : Trimino), ∃! (x y z : ChessboardWithCornerRemoved), 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    t x = some p ∧ t y = some p ∧ t z = some p

/-- Theorem stating that no valid tiling exists -/
theorem no_valid_tiling : ¬∃ (t : Tiling), is_valid_tiling t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l2239_223974


namespace NUMINAMATH_CALUDE_simplify_fraction_l2239_223947

theorem simplify_fraction (b : ℚ) (h : b = 2) : (15 * b^4) / (75 * b^3) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2239_223947


namespace NUMINAMATH_CALUDE_solution_x_l2239_223963

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l2239_223963


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2239_223980

/-- Calculates the molecular weight of a compound given its atomic composition and atomic weights -/
def molecular_weight (num_C num_H num_O num_N num_S : ℕ) 
                     (weight_C weight_H weight_O weight_N weight_S : ℝ) : ℝ :=
  (num_C : ℝ) * weight_C + 
  (num_H : ℝ) * weight_H + 
  (num_O : ℝ) * weight_O + 
  (num_N : ℝ) * weight_N + 
  (num_S : ℝ) * weight_S

/-- The molecular weight of the given compound is approximately 134.184 g/mol -/
theorem compound_molecular_weight : 
  ∀ (ε : ℝ), ε > 0 → 
  |molecular_weight 4 8 2 1 1 12.01 1.008 16.00 14.01 32.07 - 134.184| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2239_223980


namespace NUMINAMATH_CALUDE_morning_eggs_count_l2239_223967

/-- The number of eggs used in a day at the Wafting Pie Company -/
def total_eggs : ℕ := 1339

/-- The number of eggs used in the afternoon at the Wafting Pie Company -/
def afternoon_eggs : ℕ := 523

/-- The number of eggs used in the morning at the Wafting Pie Company -/
def morning_eggs : ℕ := total_eggs - afternoon_eggs

theorem morning_eggs_count : morning_eggs = 816 := by sorry

end NUMINAMATH_CALUDE_morning_eggs_count_l2239_223967


namespace NUMINAMATH_CALUDE_only_one_chooses_course_a_l2239_223926

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of course selection combinations -/
def total_combinations (n k : ℕ) : ℕ := (choose n k) * (choose n k)

/-- The number of combinations where both people choose course A -/
def both_choose_a (n k : ℕ) : ℕ := (choose (n - 1) (k - 1)) * (choose (n - 1) (k - 1))

/-- The number of ways in which only one person chooses course A -/
def only_one_chooses_a (n k : ℕ) : ℕ := (total_combinations n k) - (both_choose_a n k)

theorem only_one_chooses_course_a :
  only_one_chooses_a 4 2 = 27 := by sorry

end NUMINAMATH_CALUDE_only_one_chooses_course_a_l2239_223926


namespace NUMINAMATH_CALUDE_bounded_sequence_l2239_223997

/-- A sequence defined recursively with a parameter c -/
def x (c : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => (x c n)^2 + c

/-- The theorem stating the condition for boundedness of the sequence -/
theorem bounded_sequence (c : ℝ) (h : c > 0) :
  (∀ n, |x c n| < 2016) ↔ c ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bounded_sequence_l2239_223997


namespace NUMINAMATH_CALUDE_mike_ride_distance_l2239_223943

-- Define the taxi fare structure for each route
structure TaxiFare :=
  (initial_fee : ℚ)
  (per_mile_rate : ℚ)
  (extra_fee : ℚ)

-- Define the routes
def route_a : TaxiFare := ⟨2.5, 0.25, 3⟩
def route_b : TaxiFare := ⟨2.5, 0.3, 4⟩
def route_c : TaxiFare := ⟨2.5, 0.25, 9⟩ -- Combined bridge toll and traffic surcharge

-- Calculate the fare for a given route and distance
def calculate_fare (route : TaxiFare) (miles : ℚ) : ℚ :=
  route.initial_fee + route.per_mile_rate * miles + route.extra_fee

-- Theorem statement
theorem mike_ride_distance :
  let annie_miles : ℚ := 14
  let annie_fare := calculate_fare route_c annie_miles
  ∃ (mike_miles : ℚ), 
    (calculate_fare route_a mike_miles = annie_fare) ∧
    (mike_miles = 38) :=
by
  sorry


end NUMINAMATH_CALUDE_mike_ride_distance_l2239_223943


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l2239_223932

theorem squirrels_and_nuts :
  let num_squirrels : ℕ := 4
  let num_nuts : ℕ := 2
  num_squirrels - num_nuts = 2 :=
by sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l2239_223932


namespace NUMINAMATH_CALUDE_star_properties_l2239_223921

def star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∀ x : ℝ, star (x + 1) (x - 1) = star x x - 1) ∧
  (∀ e : ℝ, ∃ x : ℝ, star x e ≠ x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2239_223921


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2239_223961

theorem prime_sum_theorem (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2239_223961


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2239_223922

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) -- radius of cylinder and cones
  (h_cylinder : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of each cone
  (h_cylinder_eq : h_cylinder = 2 * h_cone) -- cylinder height is twice the cone height
  (r_eq : r = 10) -- radius is 10 cm
  (h_cone_eq : h_cone = 15) -- cone height is 15 cm
  : π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2239_223922


namespace NUMINAMATH_CALUDE_blue_green_difference_l2239_223931

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles + 18,
    green_tiles := figure.green_tiles + 18 }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { blue_tiles := 15, green_tiles := 9 }

/-- The new figure after adding both borders -/
def new_figure : HexagonalFigure :=
  add_border (add_border initial_figure)

theorem blue_green_difference :
  new_figure.blue_tiles - new_figure.green_tiles = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_green_difference_l2239_223931


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2239_223964

theorem coin_flip_probability (n : ℕ) : n = 7 → (n.choose 2 : ℚ) / 2^n = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2239_223964


namespace NUMINAMATH_CALUDE_school_trip_combinations_l2239_223914

/-- The number of different combinations of riding groups and ride choices -/
def ride_combinations (total_people : ℕ) (group_size : ℕ) (ride_choices : ℕ) : ℕ :=
  Nat.choose total_people group_size * ride_choices

/-- Theorem: Given 8 people, rides of 4, and 2 choices, there are 140 combinations -/
theorem school_trip_combinations :
  ride_combinations 8 4 2 = 140 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_combinations_l2239_223914


namespace NUMINAMATH_CALUDE_classroom_discussion_group_l2239_223911

def group_sizes : List Nat := [2, 3, 5, 6, 7, 8, 11, 12, 13, 17, 20, 22, 24]

theorem classroom_discussion_group (
  total_groups : Nat) 
  (lecture_groups : Nat) 
  (chinese_lecture_ratio : Nat) 
  (h1 : total_groups = 13)
  (h2 : lecture_groups = 12)
  (h3 : chinese_lecture_ratio = 6)
  (h4 : group_sizes.length = total_groups)
  (h5 : group_sizes.sum = 150) :
  ∃ x : Nat, x ∈ group_sizes ∧ x % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_discussion_group_l2239_223911


namespace NUMINAMATH_CALUDE_lcm_theorem_l2239_223936

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def lcm_condition (ab cd : ℕ) : Prop :=
  is_two_digit ab ∧ is_two_digit cd ∧
  Nat.lcm ab cd = (7 * Nat.lcm (reverse_digits ab) (reverse_digits cd)) / 4

theorem lcm_theorem (ab cd : ℕ) (h : lcm_condition ab cd) :
  Nat.lcm ab cd = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_theorem_l2239_223936


namespace NUMINAMATH_CALUDE_f_range_l2239_223970

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.log (1 + x) + x^2
  else -x * Real.log (1 - x) + x^2

theorem f_range (a : ℝ) : f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2239_223970


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l2239_223903

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l2239_223903


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2239_223966

def A : Set ℤ := {0, -2}
def B : Set ℤ := {-4, 0}

theorem union_of_A_and_B :
  A ∪ B = {-4, -2, 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2239_223966


namespace NUMINAMATH_CALUDE_f_positive_implies_a_greater_than_half_open_l2239_223924

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

-- State the theorem
theorem f_positive_implies_a_greater_than_half_open :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x > 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_greater_than_half_open_l2239_223924


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2239_223935

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 20π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 4√5. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 20 * π) →
  ∃ (c : ℝ), c^2 = 80 ∧ c = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2239_223935


namespace NUMINAMATH_CALUDE_diana_hits_eight_l2239_223958

structure Friend where
  name : String
  score : Nat

def target_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def friends : List Friend := [
  { name := "Alex", score := 18 },
  { name := "Betsy", score := 5 },
  { name := "Carlos", score := 12 },
  { name := "Diana", score := 14 },
  { name := "Edward", score := 19 },
  { name := "Fiona", score := 11 }
]

theorem diana_hits_eight :
  ∀ (assignments : List (List Nat)),
    (∀ f : Friend, f ∈ friends → 
      ∃! pair : List Nat, pair ∈ assignments ∧ pair.length = 2 ∧ pair.sum = f.score) →
    (∀ pair : List Nat, pair ∈ assignments → 
      pair.length = 2 ∧ pair.toFinset ⊆ target_scores.toFinset) →
    (∀ n : Nat, n ∈ target_scores → 
      (assignments.join.count n ≤ 1)) →
    ∃ pair : List Nat, pair ∈ assignments ∧ 
      pair.length = 2 ∧ 
      pair.sum = 14 ∧ 
      8 ∈ pair :=
by sorry

end NUMINAMATH_CALUDE_diana_hits_eight_l2239_223958


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l2239_223939

theorem difference_of_squares_form (x y : ℝ) : 
  ∃ (a b : ℝ), (-x + y) * (x + y) = a^2 - b^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l2239_223939


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_is_eight_l2239_223983

/-- The minimum number of employees with birthdays on Wednesday -/
def min_wednesday_birthdays (total_employees : ℕ) (days_in_week : ℕ) : ℕ :=
  let other_days := days_in_week - 1
  let max_other_day_birthdays := (total_employees - 1) / days_in_week
  max_other_day_birthdays + 1

/-- Prove that given 50 employees, excluding those born in March, and with Wednesday having more 
    birthdays than any other day of the week (which all have an equal number of birthdays), 
    the minimum number of employees having birthdays on Wednesday is 8. -/
theorem min_wednesday_birthdays_is_eight :
  min_wednesday_birthdays 50 7 = 8 := by
  sorry

#eval min_wednesday_birthdays 50 7

end NUMINAMATH_CALUDE_min_wednesday_birthdays_is_eight_l2239_223983


namespace NUMINAMATH_CALUDE_point_on_line_l2239_223927

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)}

/-- The problem statement -/
theorem point_on_line :
  let p1 : Point := ⟨0, 10⟩
  let p2 : Point := ⟨5, 0⟩
  let p3 : Point := ⟨x, -5⟩
  p3 ∈ Line p1 p2 → x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2239_223927


namespace NUMINAMATH_CALUDE_aqua_park_earnings_l2239_223940

/-- Calculate the total earnings of an aqua park given admission cost, tour cost, and group sizes. -/
theorem aqua_park_earnings
  (admission_cost : ℕ)
  (tour_cost : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h1 : admission_cost = 12)
  (h2 : tour_cost = 6)
  (h3 : group1_size = 10)
  (h4 : group2_size = 5) :
  (group1_size * (admission_cost + tour_cost)) + (group2_size * admission_cost) = 240 := by
  sorry

#check aqua_park_earnings

end NUMINAMATH_CALUDE_aqua_park_earnings_l2239_223940


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2239_223960

theorem negation_of_proposition (p : Prop) :
  (p ↔ ∃ x, x < 1 ∧ x^2 ≤ 1) →
  (¬p ↔ ∀ x, x < 1 → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2239_223960


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l2239_223984

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l2239_223984


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l2239_223954

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_on_third_day (total_tickets : ℕ) (friends : ℕ) (tickets_per_friend : ℕ) (second_day_sales : ℕ) : ℕ :=
  total_tickets - (friends * tickets_per_friend + second_day_sales)

/-- Theorem stating the number of tickets Amanda needs to sell on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_on_third_day 80 5 4 32 = 28 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l2239_223954


namespace NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l2239_223944

theorem min_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
by sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l2239_223944


namespace NUMINAMATH_CALUDE_transylvanian_logic_l2239_223941

/-- Represents the possible types of beings in Transylvania -/
inductive Being
| Human
| Vampire

/-- Represents the possible responses to questions -/
inductive Response
| Yes
| No

/-- A function that determines how a being responds to a question about another being's type -/
def respond (respondent : Being) (subject : Being) : Response :=
  match respondent, subject with
  | Being.Human, Being.Human => Response.Yes
  | Being.Human, Being.Vampire => Response.No
  | Being.Vampire, Being.Human => Response.No
  | Being.Vampire, Being.Vampire => Response.Yes

theorem transylvanian_logic (A B : Being) 
  (h1 : respond A B = Response.Yes) : 
  respond B A = Response.Yes := by
  sorry

end NUMINAMATH_CALUDE_transylvanian_logic_l2239_223941


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2239_223928

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point A
def A : ℝ × ℝ := (0, 1)

-- Define the condition for a point to be on a line passing through A
def on_line_through_A (k b x y : ℝ) : Prop := y = k * x + b ∧ b ≠ 1

-- Define the perpendicular condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 0) * (x₂ - 0) + (y₁ - 1) * (y₂ - 1) = 0

-- Main theorem
theorem line_passes_through_fixed_point 
  (k b x₁ y₁ x₂ y₂ : ℝ) :
  C x₁ y₁ → C x₂ y₂ → 
  on_line_through_A k b x₁ y₁ → on_line_through_A k b x₂ y₂ →
  perpendicular_condition x₁ y₁ x₂ y₂ →
  b = -3/5 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2239_223928


namespace NUMINAMATH_CALUDE_alices_june_burger_spending_l2239_223978

/-- Calculate Alice's spending on burgers in June --/
def alices_burger_spending (
  days_in_june : Nat)
  (burgers_per_day : Nat)
  (burger_price : ℚ)
  (discount_days : Nat)
  (discount_percentage : ℚ)
  (free_burger_days : Nat)
  (coupon_count : Nat)
  (coupon_discount : ℚ) : ℚ :=
  let total_burgers := days_in_june * burgers_per_day
  let regular_cost := total_burgers * burger_price
  let discount_burgers := discount_days * burgers_per_day
  let discount_amount := discount_burgers * burger_price * discount_percentage
  let free_burgers := free_burger_days
  let free_burger_value := free_burgers * burger_price
  let coupon_savings := coupon_count * burger_price * coupon_discount
  regular_cost - discount_amount - free_burger_value - coupon_savings

/-- Theorem stating Alice's spending on burgers in June --/
theorem alices_june_burger_spending :
  alices_burger_spending 30 4 13 8 (1/10) 4 6 (1/2) = 1146.6 := by
  sorry

end NUMINAMATH_CALUDE_alices_june_burger_spending_l2239_223978


namespace NUMINAMATH_CALUDE_area_between_curves_l2239_223998

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0)..(1), (f x - g x) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l2239_223998


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l2239_223905

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 11) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 1296/11 := by
  sorry

theorem min_value_sum_reciprocals_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ p q r s t u : ℝ, 
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧
    p + q + r + s + t + u = 11 ∧
    1/p + 9/q + 25/r + 49/s + 81/t + 121/u < 1296/11 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l2239_223905


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2239_223990

theorem quadratic_roots_range (m l : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2*x + l = 0 ∧ m * y^2 - 2*y + l = 0) → 
  (0 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2239_223990


namespace NUMINAMATH_CALUDE_fly_distance_bounded_l2239_223972

/-- Represents a right triangle room -/
structure RightTriangleRoom where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- Represents a fly's path in the room -/
structure FlyPath where
  room : RightTriangleRoom
  num_turns : ℕ
  start_acute_angle : Bool

/-- The maximum distance a fly can travel in the room -/
noncomputable def max_fly_distance (path : FlyPath) : ℝ :=
  sorry

/-- Theorem stating that a fly cannot travel more than 10 meters in the given conditions -/
theorem fly_distance_bounded (path : FlyPath) 
  (h1 : path.room.hypotenuse = 5)
  (h2 : path.num_turns = 10)
  (h3 : path.start_acute_angle = true) : 
  max_fly_distance path ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_fly_distance_bounded_l2239_223972


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l2239_223965

/-- 
Given a trader who sells cloth with the following conditions:
- total_meters: The total number of meters of cloth sold
- selling_price: The total selling price for all meters of cloth
- profit_per_meter: The profit made per meter of cloth

This theorem proves that the cost price per meter of cloth is equal to
(selling_price - (total_meters * profit_per_meter)) / total_meters
-/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price profit_per_meter : ℚ) 
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 5) :
  (selling_price - (total_meters : ℚ) * profit_per_meter) / total_meters = 100 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l2239_223965


namespace NUMINAMATH_CALUDE_cube_of_negative_l2239_223995

theorem cube_of_negative (a : ℝ) : (-a)^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l2239_223995


namespace NUMINAMATH_CALUDE_fraction_calculation_l2239_223971

theorem fraction_calculation : (5 / 6 : ℚ) * (1 / ((7 / 8 : ℚ) - (3 / 4 : ℚ))) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2239_223971


namespace NUMINAMATH_CALUDE_truck_tunnel_height_l2239_223946

theorem truck_tunnel_height (tunnel_radius : ℝ) (truck_width : ℝ) 
  (h_radius : tunnel_radius = 4.5)
  (h_width : truck_width = 2.7) :
  Real.sqrt (tunnel_radius^2 - (truck_width/2)^2) = 3.6 := by
sorry

end NUMINAMATH_CALUDE_truck_tunnel_height_l2239_223946


namespace NUMINAMATH_CALUDE_root_equation_solution_l2239_223956

theorem root_equation_solution (a : ℚ) : 
  ((-2 : ℚ)^2 - a * (-2) + 7 = 0) → a = -11/2 := by
sorry

end NUMINAMATH_CALUDE_root_equation_solution_l2239_223956


namespace NUMINAMATH_CALUDE_badge_exchange_problem_l2239_223917

theorem badge_exchange_problem (vasya_initial tolya_initial : ℕ) : 
  vasya_initial = 50 ∧ tolya_initial = 45 →
  vasya_initial = tolya_initial + 5 ∧
  (vasya_initial - (vasya_initial * 24 / 100) + (tolya_initial * 20 / 100)) + 1 =
  (tolya_initial - (tolya_initial * 20 / 100) + (vasya_initial * 24 / 100)) :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_problem_l2239_223917


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2239_223979

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2239_223979


namespace NUMINAMATH_CALUDE_cube_congruence_l2239_223930

theorem cube_congruence (a b : ℕ) : a ≡ b [MOD 1000] → a^3 ≡ b^3 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_cube_congruence_l2239_223930


namespace NUMINAMATH_CALUDE_complex_fraction_calculations_l2239_223910

theorem complex_fraction_calculations :
  (1 / 60) / ((1 / 3) - (1 / 4) + (1 / 12)) = 1 / 10 ∧
  -(1 / 42) / ((3 / 7) - (5 / 14) + (2 / 3) - (1 / 6)) = -(1 / 24) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculations_l2239_223910


namespace NUMINAMATH_CALUDE_ab_value_l2239_223969

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2239_223969


namespace NUMINAMATH_CALUDE_johns_weekly_earnings_l2239_223977

/-- Calculates John's total earnings per week from crab fishing --/
theorem johns_weekly_earnings :
  let monday_baskets : ℕ := 3
  let thursday_baskets : ℕ := 4
  let small_crabs_per_basket : ℕ := 4
  let large_crabs_per_basket : ℕ := 5
  let small_crab_price : ℕ := 3
  let large_crab_price : ℕ := 5

  let monday_crabs : ℕ := monday_baskets * small_crabs_per_basket
  let thursday_crabs : ℕ := thursday_baskets * large_crabs_per_basket

  let monday_earnings : ℕ := monday_crabs * small_crab_price
  let thursday_earnings : ℕ := thursday_crabs * large_crab_price

  let total_earnings : ℕ := monday_earnings + thursday_earnings

  total_earnings = 136 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_earnings_l2239_223977


namespace NUMINAMATH_CALUDE_total_drying_time_in_hours_l2239_223975

/-- Time to dry a short-haired dog in minutes -/
def short_hair_time : ℕ := 10

/-- Time to dry a full-haired dog in minutes -/
def full_hair_time : ℕ := 2 * short_hair_time

/-- Time to dry a medium-haired dog in minutes -/
def medium_hair_time : ℕ := 15

/-- Number of short-haired dogs -/
def short_hair_count : ℕ := 12

/-- Number of full-haired dogs -/
def full_hair_count : ℕ := 15

/-- Number of medium-haired dogs -/
def medium_hair_count : ℕ := 8

/-- Total time to dry all dogs in minutes -/
def total_time : ℕ := 
  short_hair_time * short_hair_count + 
  full_hair_time * full_hair_count + 
  medium_hair_time * medium_hair_count

theorem total_drying_time_in_hours : 
  total_time / 60 = 9 := by sorry

end NUMINAMATH_CALUDE_total_drying_time_in_hours_l2239_223975


namespace NUMINAMATH_CALUDE_rational_power_equality_l2239_223908

theorem rational_power_equality (x y : ℚ) (n : ℕ) (h_odd : Odd n) (h_pos : 0 < n)
  (h_eq : x^n - 2*x = y^n - 2*y) : x = y := by
  sorry

end NUMINAMATH_CALUDE_rational_power_equality_l2239_223908


namespace NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l2239_223989

theorem greatest_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 15 ≠ -6) ↔ b ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l2239_223989


namespace NUMINAMATH_CALUDE_false_proposition_implies_plane_plane_line_l2239_223991

-- Define geometric figures
inductive GeometricFigure
  | Line
  | Plane

-- Define perpendicular and parallel relations
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  perpendicular x y → parallel y z → perpendicular x z

-- Theorem statement
theorem false_proposition_implies_plane_plane_line :
  ∀ x y z : GeometricFigure,
  ¬(proposition x y z) →
  (x = GeometricFigure.Plane ∧ y = GeometricFigure.Plane ∧ z = GeometricFigure.Line) :=
sorry

end NUMINAMATH_CALUDE_false_proposition_implies_plane_plane_line_l2239_223991


namespace NUMINAMATH_CALUDE_unique_two_digit_number_with_reverse_difference_64_l2239_223909

theorem unique_two_digit_number_with_reverse_difference_64 :
  ∃! N : ℕ, 
    (N ≥ 10 ∧ N < 100) ∧ 
    (∃ a : ℕ, a < 10 ∧ N = 10 * a + 1) ∧
    ((10 * (N % 10) + N / 10) - N = 64) := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_with_reverse_difference_64_l2239_223909


namespace NUMINAMATH_CALUDE_four_letter_initials_count_l2239_223952

theorem four_letter_initials_count : 
  let letter_count : ℕ := 10
  let initial_length : ℕ := 4
  let order_matters : Bool := true
  let allow_repetition : Bool := true
  (letter_count ^ initial_length : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_four_letter_initials_count_l2239_223952


namespace NUMINAMATH_CALUDE_height_weight_relationship_l2239_223912

/-- Represents the coefficient of determination (R²) in a linear regression model -/
def R_squared : ℝ := 0.64

/-- The proportion of variation explained by the model -/
def variation_explained : ℝ := R_squared

/-- The proportion of variation not explained by the model (random error) -/
def variation_unexplained : ℝ := 1 - R_squared

theorem height_weight_relationship :
  variation_explained = 0.64 ∧
  variation_unexplained = 0.36 ∧
  variation_explained + variation_unexplained = 1 := by
  sorry

#eval R_squared
#eval variation_explained
#eval variation_unexplained

end NUMINAMATH_CALUDE_height_weight_relationship_l2239_223912


namespace NUMINAMATH_CALUDE_fair_tickets_sold_l2239_223981

theorem fair_tickets_sold (total : ℕ) (second_week : ℕ) (left_to_sell : ℕ) 
  (h1 : total = 90)
  (h2 : second_week = 17)
  (h3 : left_to_sell = 35) :
  total - second_week - left_to_sell = 38 := by
sorry

end NUMINAMATH_CALUDE_fair_tickets_sold_l2239_223981


namespace NUMINAMATH_CALUDE_policeman_can_reach_gangster_side_l2239_223992

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  center : Point
  vertex : Point

/-- Represents the maximum speeds of the policeman and gangster -/
structure Speeds where
  policeman : ℝ
  gangster : ℝ

/-- Theorem stating that the policeman can always reach the same side as the gangster -/
theorem policeman_can_reach_gangster_side (s : ℝ) (square : Square s) (speeds : Speeds) :
  s > 0 ∧
  square.center = Point.mk (s/2) (s/2) ∧
  (square.vertex = Point.mk 0 0 ∨ square.vertex = Point.mk s 0 ∨
   square.vertex = Point.mk 0 s ∨ square.vertex = Point.mk s s) ∧
  speeds.gangster = 2.9 * speeds.policeman →
  ∃ (t : ℝ), t > 0 ∧ 
    ∃ (p : Point), (p.x = 0 ∨ p.x = s ∨ p.y = 0 ∨ p.y = s) ∧
      (p.x - square.center.x)^2 + (p.y - square.center.y)^2 ≤ (speeds.policeman * t)^2 ∧
      ((p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 ≤ (speeds.gangster * t)^2 ∨
       (p.x - square.vertex.x)^2 + (p.y - square.vertex.y)^2 = (s * speeds.gangster * t)^2) :=
by sorry

end NUMINAMATH_CALUDE_policeman_can_reach_gangster_side_l2239_223992


namespace NUMINAMATH_CALUDE_arcsin_of_negative_one_l2239_223999

theorem arcsin_of_negative_one :
  Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_negative_one_l2239_223999


namespace NUMINAMATH_CALUDE_mean_proportional_sum_l2239_223942

/-- Mean proportional of two numbers -/
def mean_proportional (a b c : ℝ) : Prop := a / b = b / c

/-- Find x such that 0.9 : 0.6 = 0.6 : x -/
def find_x : ℝ := 0.4

/-- Find y such that 1/2 : 1/5 = 1/5 : y -/
def find_y : ℝ := 0.08

theorem mean_proportional_sum :
  mean_proportional 0.9 0.6 find_x ∧ 
  mean_proportional (1/2) (1/5) find_y ∧
  find_x + find_y = 0.48 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_sum_l2239_223942


namespace NUMINAMATH_CALUDE_union_cardinality_l2239_223962

def A : Finset ℕ := {4, 5, 7, 9}
def B : Finset ℕ := {3, 4, 7, 8, 9}

theorem union_cardinality : (A ∪ B).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_l2239_223962


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l2239_223915

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    interior_angle = (n - 2) * 180 / n →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l2239_223915


namespace NUMINAMATH_CALUDE_f_is_odd_g_sum_one_l2239_223973

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the given conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_one_zero : f 1 = 0

-- State the theorems to be proved
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem g_sum_one : g 1 + g (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_f_is_odd_g_sum_one_l2239_223973


namespace NUMINAMATH_CALUDE_vertex_angle_is_160_degrees_l2239_223976

/-- An isosceles triangle with specific properties -/
structure SpecialIsoscelesTriangle where
  -- The length of each equal side
  a : ℝ
  -- The base of the triangle
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The vertex angle in radians
  θ : ℝ
  -- The triangle is isosceles
  isIsosceles : b = 2 * a * Real.cos θ
  -- The square of the length of each equal side is three times the product of the base and the height
  sideSquareProperty : a^2 = 3 * b * h
  -- The triangle is obtuse
  isObtuse : θ > Real.pi / 2

/-- The theorem stating that the vertex angle of the special isosceles triangle is 160 degrees -/
theorem vertex_angle_is_160_degrees (t : SpecialIsoscelesTriangle) : 
  t.θ = 160 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_vertex_angle_is_160_degrees_l2239_223976


namespace NUMINAMATH_CALUDE_darryl_honeydew_price_l2239_223920

/-- The price of a honeydew given Darryl's sales data -/
def honeydew_price (cantaloupe_price : ℚ) (initial_cantaloupes : ℕ) (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ) (rotten_honeydews : ℕ) (final_cantaloupes : ℕ) (final_honeydews : ℕ)
  (total_revenue : ℚ) : ℚ :=
  let sold_cantaloupes := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews := initial_honeydews - final_honeydews - rotten_honeydews
  let cantaloupe_revenue := cantaloupe_price * sold_cantaloupes
  let honeydew_revenue := total_revenue - cantaloupe_revenue
  honeydew_revenue / sold_honeydews

theorem darryl_honeydew_price :
  honeydew_price 2 30 27 2 3 8 9 85 = 3 := by
  sorry

#eval honeydew_price 2 30 27 2 3 8 9 85

end NUMINAMATH_CALUDE_darryl_honeydew_price_l2239_223920


namespace NUMINAMATH_CALUDE_files_left_theorem_l2239_223919

/-- Calculates the number of files left after deletion -/
def files_left (initial_files : ℕ) (deleted_files : ℕ) : ℕ :=
  initial_files - deleted_files

/-- Theorem: The number of files left is the difference between initial files and deleted files -/
theorem files_left_theorem (initial_files deleted_files : ℕ) 
  (h : deleted_files ≤ initial_files) : 
  files_left initial_files deleted_files = initial_files - deleted_files :=
by
  sorry

#eval files_left 21 14  -- Should output 7

end NUMINAMATH_CALUDE_files_left_theorem_l2239_223919


namespace NUMINAMATH_CALUDE_empty_carton_weight_l2239_223953

/-- Given the weights of a half-full and full milk carton, calculate the weight of an empty carton -/
theorem empty_carton_weight (half_full_weight full_weight : ℝ) :
  half_full_weight = 5 →
  full_weight = 8 →
  full_weight - 2 * (full_weight - half_full_weight) = 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_carton_weight_l2239_223953


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2239_223900

theorem quadratic_roots_problem (a b n r s : ℝ) : 
  a^2 - n*a + 6 = 0 →
  b^2 - n*b + 6 = 0 →
  (a + 1/b)^2 - r*(a + 1/b) + s = 0 →
  (b + 1/a)^2 - r*(b + 1/a) + s = 0 →
  s = 49/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2239_223900


namespace NUMINAMATH_CALUDE_largest_multiple_80_correct_l2239_223929

/-- Returns true if all digits of n are either 8 or 0 -/
def allDigits80 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest multiple of 20 with all digits 8 or 0 -/
def largestMultiple80 : ℕ := 8880

theorem largest_multiple_80_correct :
  largestMultiple80 % 20 = 0 ∧
  allDigits80 largestMultiple80 ∧
  ∀ n : ℕ, n > largestMultiple80 → ¬(n % 20 = 0 ∧ allDigits80 n) :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_80_correct_l2239_223929


namespace NUMINAMATH_CALUDE_solution_set_correct_l2239_223994

/-- The system of equations --/
def system (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

/-- The solution set --/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (2, 1, 3), (2, -1, -3), (-2, 1, -3), (-2, -1, 3)}

/-- Theorem stating that the solution set is correct --/
theorem solution_set_correct :
  ∀ x y z, (x, y, z) ∈ solution_set ↔ system x y z :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2239_223994


namespace NUMINAMATH_CALUDE_det_3A_eq_96_l2239_223951

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -5, 6]

theorem det_3A_eq_96 : Matrix.det (3 • A) = 96 := by
  sorry

end NUMINAMATH_CALUDE_det_3A_eq_96_l2239_223951


namespace NUMINAMATH_CALUDE_infinitely_many_non_squares_l2239_223904

theorem infinitely_many_non_squares (a b c : ℕ+) :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ, (n.val : ℤ)^3 + (a.val : ℤ) * (n.val : ℤ)^2 + (b.val : ℤ) * (n.val : ℤ) + (c.val : ℤ) ≠ (k : ℤ)^2} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_non_squares_l2239_223904


namespace NUMINAMATH_CALUDE_segment_length_is_twenty_l2239_223923

/-- The volume of a geometric body formed by points whose distance to a line segment
    is no greater than r units -/
noncomputable def geometricBodyVolume (r : ℝ) (segmentLength : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3 + Real.pi * r^2 * segmentLength

/-- Theorem stating that if the volume of the geometric body with radius 3
    is 216π, then the segment length is 20 -/
theorem segment_length_is_twenty (segmentLength : ℝ) :
  geometricBodyVolume 3 segmentLength = 216 * Real.pi → segmentLength = 20 := by
  sorry

#check segment_length_is_twenty

end NUMINAMATH_CALUDE_segment_length_is_twenty_l2239_223923


namespace NUMINAMATH_CALUDE_room_occupancy_l2239_223918

theorem room_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_chairs / 4 = total_chairs - 6) →  -- Three-fourths of chairs are occupied
  (2 * total_people / 3 = 3 * total_chairs / 4) →  -- Two-thirds of people are seated
  total_people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l2239_223918


namespace NUMINAMATH_CALUDE_function_cycle_existence_l2239_223950

theorem function_cycle_existence :
  ∃ (f : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ),
    (∃ (a b c d : ℝ), ∀ x, f x = (a * x + b) / (c * x + d)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
    x₄ ≠ x₅ ∧
    f x₁ = x₂ ∧ f x₂ = x₃ ∧ f x₃ = x₄ ∧ f x₄ = x₅ ∧ f x₅ = x₁ := by
  sorry

end NUMINAMATH_CALUDE_function_cycle_existence_l2239_223950


namespace NUMINAMATH_CALUDE_min_floor_sum_l2239_223993

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = a*b*c) : 
  ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l2239_223993


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2239_223959

theorem rationalize_denominator : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2239_223959


namespace NUMINAMATH_CALUDE_even_red_faces_count_l2239_223957

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of red faces in a painted block -/
def countEvenRedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x4x2 block has 24 cubes with an even number of red faces -/
theorem even_red_faces_count (b : Block) (h1 : b.length = 6) (h2 : b.width = 4) (h3 : b.height = 2) :
  countEvenRedFaces b = 24 := by
  sorry

#check even_red_faces_count

end NUMINAMATH_CALUDE_even_red_faces_count_l2239_223957


namespace NUMINAMATH_CALUDE_min_value_inequality_l2239_223913

theorem min_value_inequality (x y z : ℝ) (h : x + 2*y + 3*z = 1) : 
  x^2 + 2*y^2 + 3*z^2 ≥ 1/3 := by
  sorry

#check min_value_inequality

end NUMINAMATH_CALUDE_min_value_inequality_l2239_223913


namespace NUMINAMATH_CALUDE_arctan_sum_not_standard_angle_l2239_223933

theorem arctan_sum_not_standard_angle :
  let a : ℝ := 2/3
  let b : ℝ := (3 / (5/3)) - 1
  ¬(Real.arctan a + Real.arctan b = π/2 ∨
    Real.arctan a + Real.arctan b = π/3 ∨
    Real.arctan a + Real.arctan b = π/4 ∨
    Real.arctan a + Real.arctan b = π/5 ∨
    Real.arctan a + Real.arctan b = π/6) :=
by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_not_standard_angle_l2239_223933


namespace NUMINAMATH_CALUDE_planting_schemes_count_l2239_223945

def number_of_seeds : ℕ := 6
def number_of_plots : ℕ := 4
def number_of_first_plot_options : ℕ := 2

def planting_schemes : ℕ :=
  number_of_first_plot_options * (number_of_seeds - 1).factorial / (number_of_seeds - number_of_plots).factorial

theorem planting_schemes_count : planting_schemes = 120 := by
  sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l2239_223945


namespace NUMINAMATH_CALUDE_workshop_nobel_laureates_l2239_223996

theorem workshop_nobel_laureates
  (total_scientists : ℕ)
  (wolf_laureates : ℕ)
  (wolf_and_nobel : ℕ)
  (h_total : total_scientists = 50)
  (h_wolf : wolf_laureates = 31)
  (h_both : wolf_and_nobel = 16)
  (h_diff : ∃ (non_nobel : ℕ), 
    wolf_laureates + non_nobel + (non_nobel + 3) = total_scientists) :
  ∃ (nobel_laureates : ℕ), 
    nobel_laureates = 27 ∧ 
    nobel_laureates ≤ total_scientists ∧
    wolf_and_nobel ≤ nobel_laureates ∧
    wolf_and_nobel ≤ wolf_laureates :=
by
  sorry


end NUMINAMATH_CALUDE_workshop_nobel_laureates_l2239_223996


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l2239_223906

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l2239_223906


namespace NUMINAMATH_CALUDE_min_sum_at_6_l2239_223938

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = -2 ∧ seq.S 20 = 16

/-- The main theorem -/
theorem min_sum_at_6 (seq : ArithmeticSequence) 
  (h : problem_conditions seq) :
  ∀ n : ℕ, n ≠ 0 → seq.S 6 ≤ seq.S n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l2239_223938


namespace NUMINAMATH_CALUDE_unique_number_with_divisor_sum_power_of_ten_l2239_223916

theorem unique_number_with_divisor_sum_power_of_ten (N : ℕ) : 
  (∃ m : ℕ, m < N ∧ m ∣ N ∧ (∀ d : ℕ, d < N → d ∣ N → d ≤ m) ∧ 
   (∃ k : ℕ, N + m = 10^k)) → N = 75 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_divisor_sum_power_of_ten_l2239_223916


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2239_223907

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (8 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 90 / 37 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2239_223907


namespace NUMINAMATH_CALUDE_complex_number_equality_l2239_223925

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 → (2 - i) * i = -1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2239_223925


namespace NUMINAMATH_CALUDE_elephant_count_l2239_223948

/-- The number of elephants at We Preserve For Future park -/
def W : ℕ := 70

/-- The number of elephants at Gestures For Good park -/
def G : ℕ := 3 * W

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := W + G

theorem elephant_count : total_elephants = 280 := by
  sorry

end NUMINAMATH_CALUDE_elephant_count_l2239_223948


namespace NUMINAMATH_CALUDE_acute_triangle_perimeter_bound_l2239_223986

/-- Given an acute-angled triangle with circumradius R and perimeter P, prove that P ≥ 4R. -/
theorem acute_triangle_perimeter_bound (R : ℝ) (P : ℝ) (α β γ : ℝ) :
  R > 0 →  -- R is positive (implied by being a radius)
  P > 0 →  -- P is positive (implied by being a perimeter)
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  0 < γ ∧ γ < π/2 →  -- γ is acute
  α + β + γ = π →  -- sum of angles in a triangle
  P = 2 * R * (Real.sin α + Real.sin β + Real.sin γ) →  -- perimeter formula using sine rule
  P ≥ 4 * R :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_perimeter_bound_l2239_223986


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l2239_223937

/-- The polynomial (x-1)(x+3)(x-4)(x-8)+m is a perfect square if and only if m = 196 -/
theorem polynomial_perfect_square (x m : ℝ) : 
  ∃ y : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = y^2 ↔ m = 196 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l2239_223937


namespace NUMINAMATH_CALUDE_multiple_births_l2239_223934

theorem multiple_births (total_babies : ℕ) (twins triplets quintuplets : ℕ) : 
  total_babies = 1200 →
  triplets = 2 * quintuplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 316 :=
by
  sorry

end NUMINAMATH_CALUDE_multiple_births_l2239_223934


namespace NUMINAMATH_CALUDE_min_removed_length_345_square_l2239_223949

/-- Represents a right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right_angled : a^2 + b^2 = c^2

/-- Represents a square formed by four right-angled triangles -/
structure TriangleSquare where
  triangle : RightTriangle
  side_length : ℕ
  is_valid : side_length = triangle.a + triangle.b

/-- The minimum length of line segments to be removed to make the figure drawable in one stroke -/
def min_removed_length (square : TriangleSquare) : ℕ := sorry

/-- Theorem stating that the minimum length of removed line segments is 7 for a square formed by four 3-4-5 triangles -/
theorem min_removed_length_345_square :
  ∀ (square : TriangleSquare),
    square.triangle = { a := 3, b := 4, c := 5, is_right_angled := by norm_num }
    → min_removed_length square = 7 := by sorry

end NUMINAMATH_CALUDE_min_removed_length_345_square_l2239_223949


namespace NUMINAMATH_CALUDE_vegetable_price_calculation_l2239_223901

/-- The price of vegetables and the final cost after discount -/
theorem vegetable_price_calculation :
  let cucumber_price : ℝ := 5
  let tomato_price : ℝ := cucumber_price * 0.8
  let bell_pepper_price : ℝ := cucumber_price * 1.5
  let total_cost : ℝ := 2 * tomato_price + 3 * cucumber_price + 4 * bell_pepper_price
  let discount_rate : ℝ := 0.1
  let final_price : ℝ := total_cost * (1 - discount_rate)
  final_price = 47.7 := by
sorry


end NUMINAMATH_CALUDE_vegetable_price_calculation_l2239_223901


namespace NUMINAMATH_CALUDE_all_equations_have_integer_roots_l2239_223968

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic equation has integer roots -/
def hasIntegerRoots (eq : QuadraticEquation) : Prop :=
  ∃ x y : ℤ, eq.a * x^2 + eq.b * x + eq.c = 0 ∧ eq.a * y^2 + eq.b * y + eq.c = 0 ∧ x ≠ y

/-- Generates the next equation by increasing coefficients by 1 -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { a := eq.a, b := eq.b + 1, c := eq.c + 1 }

/-- The initial quadratic equation x^2 + 3x + 2 = 0 -/
def initialEquation : QuadraticEquation := { a := 1, b := 3, c := 2 }

theorem all_equations_have_integer_roots :
  hasIntegerRoots initialEquation ∧
  hasIntegerRoots (nextEquation initialEquation) ∧
  hasIntegerRoots (nextEquation (nextEquation initialEquation)) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation initialEquation))) ∧
  hasIntegerRoots (nextEquation (nextEquation (nextEquation (nextEquation initialEquation)))) :=
by sorry


end NUMINAMATH_CALUDE_all_equations_have_integer_roots_l2239_223968


namespace NUMINAMATH_CALUDE_length_of_AE_l2239_223982

-- Define the square and points
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (4, 0) ∧ C = (4, 4) ∧ D = (0, 4)

def PointOnSide (E : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ E = (x, 0)

def ReflectionOverDiagonal (E F : ℝ × ℝ) : Prop :=
  F.1 + F.2 = 4 ∧ F.1 = 4 - E.1

def DistanceCondition (D E F : ℝ × ℝ) : Prop :=
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 4 * (E.1 - D.1)^2

-- Main theorem
theorem length_of_AE (A B C D E F : ℝ × ℝ) :
  Square A B C D →
  PointOnSide E →
  ReflectionOverDiagonal E F →
  DistanceCondition D E F →
  E.1 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AE_l2239_223982


namespace NUMINAMATH_CALUDE_sam_travel_time_l2239_223985

-- Define the points and distances
def point_A : ℝ := 0
def point_B : ℝ := 1000
def point_C : ℝ := 600

-- Define Sam's speed
def sam_speed : ℝ := 50

-- State the theorem
theorem sam_travel_time :
  let total_distance := point_B - point_A
  let time := total_distance / sam_speed
  (point_C - point_A = 600) ∧ 
  (point_B - point_C = 400) ∧ 
  (sam_speed = 50) →
  time = 20 := by sorry

end NUMINAMATH_CALUDE_sam_travel_time_l2239_223985


namespace NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l2239_223902

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l2239_223902


namespace NUMINAMATH_CALUDE_third_shiny_penny_probability_l2239_223987

theorem third_shiny_penny_probability :
  let total_pennies : ℕ := 9
  let shiny_pennies : ℕ := 4
  let dull_pennies : ℕ := 5
  let probability_more_than_four_draws : ℚ :=
    (Nat.choose 4 2 * Nat.choose 5 1 + Nat.choose 4 1 * Nat.choose 5 2) / Nat.choose 9 4
  probability_more_than_four_draws = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_third_shiny_penny_probability_l2239_223987


namespace NUMINAMATH_CALUDE_quadratic_equation_with_zero_root_l2239_223955

theorem quadratic_equation_with_zero_root (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + (a - 2) = 0) ∧ 
  ((a - 1) * 0^2 + 0 + (a - 2) = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_zero_root_l2239_223955
