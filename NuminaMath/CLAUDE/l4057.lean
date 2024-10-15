import Mathlib

namespace NUMINAMATH_CALUDE_books_on_cart_l4057_405717

def top_section : ℕ := 12 + 8 + 4

def bottom_section_non_mystery : ℕ := 5 + 6

def bottom_section : ℕ := 2 * bottom_section_non_mystery

def total_books : ℕ := top_section + bottom_section

theorem books_on_cart : total_books = 46 := by
  sorry

end NUMINAMATH_CALUDE_books_on_cart_l4057_405717


namespace NUMINAMATH_CALUDE_mixture_weight_l4057_405782

/-- Proves that the total weight of a mixture of cashews and peanuts is 29.5 pounds
    given specific prices and constraints. -/
theorem mixture_weight (cashew_price peanut_price total_cost cashew_weight : ℝ) 
  (h1 : cashew_price = 5)
  (h2 : peanut_price = 2)
  (h3 : total_cost = 92)
  (h4 : cashew_weight = 11) : 
  cashew_weight + (total_cost - cashew_price * cashew_weight) / peanut_price = 29.5 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l4057_405782


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_l4057_405725

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetry_about_x_axis :
  let P : Point2D := { x := -1, y := 5 }
  symmetricAboutXAxis P = { x := -1, y := -5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_l4057_405725


namespace NUMINAMATH_CALUDE_reflect_H_twice_l4057_405720

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the parallelogram EFGH
def E : Point2D := ⟨3, 6⟩
def F : Point2D := ⟨5, 10⟩
def G : Point2D := ⟨7, 6⟩
def H : Point2D := ⟨5, 2⟩

-- Define reflection across x-axis
def reflectX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

-- Define reflection across y = x + 2
def reflectYXPlus2 (p : Point2D) : Point2D :=
  ⟨p.y - 2, p.x + 2⟩

-- Theorem statement
theorem reflect_H_twice (h : Point2D) :
  h = H →
  reflectYXPlus2 (reflectX h) = ⟨-4, 7⟩ :=
by sorry

end NUMINAMATH_CALUDE_reflect_H_twice_l4057_405720


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l4057_405771

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : m + n ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l4057_405771


namespace NUMINAMATH_CALUDE_helens_hotdogs_count_l4057_405776

/-- The number of hotdogs Dylan's mother brought -/
def dylans_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := 480

/-- The number of hotdogs Helen's mother brought -/
def helens_hotdogs : ℕ := total_hotdogs - dylans_hotdogs

theorem helens_hotdogs_count : helens_hotdogs = 101 := by
  sorry

end NUMINAMATH_CALUDE_helens_hotdogs_count_l4057_405776


namespace NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l4057_405783

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_over_2_l4057_405783


namespace NUMINAMATH_CALUDE_ronald_egg_sharing_l4057_405731

def total_eggs : ℕ := 16
def eggs_per_friend : ℕ := 2

theorem ronald_egg_sharing :
  total_eggs / eggs_per_friend = 8 := by sorry

end NUMINAMATH_CALUDE_ronald_egg_sharing_l4057_405731


namespace NUMINAMATH_CALUDE_abs_minus_sqrt_eq_three_l4057_405758

theorem abs_minus_sqrt_eq_three (a : ℝ) (h : a < 0) : |a - 3| - Real.sqrt (a^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_sqrt_eq_three_l4057_405758


namespace NUMINAMATH_CALUDE_sum_of_solutions_l4057_405703

-- Define the equation
def equation (x : ℝ) : Prop := (4 * x + 6) * (3 * x - 7) = 0

-- State the theorem
theorem sum_of_solutions : 
  ∃ (s : ℝ), (∀ (x : ℝ), equation x → x = s ∨ x = (5/6 - s)) ∧ s + (5/6 - s) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l4057_405703


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l4057_405788

theorem no_square_divisible_by_six_between_50_and_120 : 
  ¬ ∃ x : ℕ, x^2 = x ∧ x % 6 = 0 ∧ 50 < x ∧ x < 120 := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l4057_405788


namespace NUMINAMATH_CALUDE_equidistant_sum_constant_sum_of_terms_l4057_405738

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of equidistant terms in an arithmetic sequence is constant -/
theorem equidistant_sum_constant {a : ℕ → ℝ} (h : arithmetic_sequence a) :
  ∀ n k : ℕ, a n + a (n + k) = a (n - 1) + a (n + k + 1) :=
sorry

theorem sum_of_terms (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end NUMINAMATH_CALUDE_equidistant_sum_constant_sum_of_terms_l4057_405738


namespace NUMINAMATH_CALUDE_range_of_a_l4057_405780

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (¬p x a → ¬q x)) →
  -1 ≤ a ∧ a ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4057_405780


namespace NUMINAMATH_CALUDE_a_range_l4057_405799

theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) → 
  3 < a ∧ a < 5 := by
sorry

end NUMINAMATH_CALUDE_a_range_l4057_405799


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l4057_405796

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ lipstick_wearers : ℕ, lipstick_wearers = total_students / 2)
  (h3 : ∃ red_lipstick_wearers : ℕ, red_lipstick_wearers = lipstick_wearers / 4)
  (h4 : ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = red_lipstick_wearers / 5) :
  ∃ blue_lipstick_wearers : ℕ, blue_lipstick_wearers = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l4057_405796


namespace NUMINAMATH_CALUDE_product_mod_600_l4057_405716

theorem product_mod_600 : (1497 * 2003) % 600 = 291 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_600_l4057_405716


namespace NUMINAMATH_CALUDE_equation_roots_l4057_405753

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 - 2*x^3 - 7*x^2 - 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 + Real.sqrt 5) / 2) ∧
    (b = (1 - Real.sqrt 5) / 2) ∧
    (c = (-1 + Real.sqrt 37) / 6) ∧
    (d = (-1 - Real.sqrt 37) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l4057_405753


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4057_405779

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 ≥ 1 ∧ x ≥ a) ↔ x ≥ 2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4057_405779


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l4057_405723

/-- Given a quadratic equation x^2 + px - q = 0 where p and q are positive real numbers,
    if the difference between its roots is 2, then p = √(4 - 4q) -/
theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let r₁ := (-p + Real.sqrt (p^2 + 4*q)) / 2
  let r₂ := (-p - Real.sqrt (p^2 + 4*q)) / 2
  (r₁ - r₂ = 2) → p = Real.sqrt (4 - 4*q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l4057_405723


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l4057_405772

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 ≡ 10 * k + 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l4057_405772


namespace NUMINAMATH_CALUDE_chess_tournament_games_l4057_405794

theorem chess_tournament_games (n : ℕ) (h : n = 14) : 
  (n.choose 2) = 91 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l4057_405794


namespace NUMINAMATH_CALUDE_hyperbola_iff_product_negative_l4057_405750

/-- Definition of a hyperbola equation -/
def is_hyperbola_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / m + (y t)^2 / n = 1 ∧
  (∃ t₁ t₂, (x t₁, y t₁) ≠ (x t₂, y t₂))

/-- The main theorem stating the condition for a hyperbola -/
theorem hyperbola_iff_product_negative (m n : ℝ) :
  is_hyperbola_equation m n ↔ m * n < 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_iff_product_negative_l4057_405750


namespace NUMINAMATH_CALUDE_stack_height_l4057_405778

/-- Calculates the vertical distance of a stack of linked rings -/
def verticalDistance (topDiameter : ℕ) (bottomDiameter : ℕ) (thickness : ℕ) : ℕ :=
  let numberOfRings := (topDiameter - bottomDiameter) / 2 + 1
  let innerDiameterSum := (numberOfRings * (topDiameter - thickness * 2 + bottomDiameter - thickness * 2)) / 2
  innerDiameterSum + thickness * 2

/-- The vertical distance of the stack of rings is 76 cm -/
theorem stack_height : verticalDistance 20 4 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_l4057_405778


namespace NUMINAMATH_CALUDE_later_arrival_l4057_405769

/-- A man's journey to his office -/
structure JourneyToOffice where
  usual_rate : ℝ
  usual_time : ℝ
  slower_rate : ℝ
  slower_time : ℝ

/-- The conditions of the problem -/
def journey_conditions (j : JourneyToOffice) : Prop :=
  j.usual_time = 1 ∧ j.slower_rate = 3/4 * j.usual_rate

/-- The theorem to be proved -/
theorem later_arrival (j : JourneyToOffice) 
  (h : journey_conditions j) : 
  j.slower_time - j.usual_time = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_later_arrival_l4057_405769


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4057_405711

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (1 : ℝ) / |x - 1| < 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 > 0}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4057_405711


namespace NUMINAMATH_CALUDE_standing_men_ratio_l4057_405759

theorem standing_men_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  (standing_men : ℚ) / (total_men : ℚ) = 1 / 8 :=
by
  intros h_total h_seated
  sorry
where
  women := (2 : ℚ) / 3 * total_passengers
  total_men := total_passengers - women
  standing_men := total_men - seated_men

end NUMINAMATH_CALUDE_standing_men_ratio_l4057_405759


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l4057_405707

theorem sqrt_sum_inequality (a b c d : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_a : a ≤ 1)
  (h_ab : a + b ≤ 5)
  (h_abc : a + b + c ≤ 14)
  (h_abcd : a + b + c + d ≤ 30) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l4057_405707


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l4057_405727

/-- Proves that the given function satisfies the differential equation -/
theorem function_satisfies_equation (x a : ℝ) :
  let y := a + (7 * x) / (a * x + 1)
  let y' := 7 / ((a * x + 1) ^ 2)
  y - x * y' = a * (1 + x^2 * y') := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l4057_405727


namespace NUMINAMATH_CALUDE_wall_thickness_calculation_l4057_405734

/-- Calculates the thickness of a wall given brick dimensions and wall specifications -/
theorem wall_thickness_calculation (brick_length brick_width brick_height : ℝ)
                                   (wall_length wall_height : ℝ)
                                   (num_bricks : ℕ) :
  brick_length = 50 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 3200 →
  ∃ (wall_thickness : ℝ),
    wall_thickness = 22.5 ∧
    wall_length * wall_height * wall_thickness = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check wall_thickness_calculation

end NUMINAMATH_CALUDE_wall_thickness_calculation_l4057_405734


namespace NUMINAMATH_CALUDE_total_rats_l4057_405749

/-- The number of rats each person has -/
structure RatCounts where
  elodie : ℕ
  hunter : ℕ
  kenia : ℕ
  teagan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (rc : RatCounts) : Prop :=
  rc.elodie = 30 ∧
  rc.hunter = rc.elodie - 10 ∧
  rc.kenia = 3 * (rc.hunter + rc.elodie) ∧
  rc.teagan = 2 * rc.elodie ∧
  rc.teagan = rc.kenia - 5

/-- The theorem stating that the total number of rats is 260 -/
theorem total_rats (rc : RatCounts) (h : satisfiesConditions rc) :
  rc.elodie + rc.hunter + rc.kenia + rc.teagan = 260 :=
by sorry

end NUMINAMATH_CALUDE_total_rats_l4057_405749


namespace NUMINAMATH_CALUDE_binomial_13_choose_10_l4057_405702

theorem binomial_13_choose_10 : Nat.choose 13 10 = 286 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_choose_10_l4057_405702


namespace NUMINAMATH_CALUDE_initial_bacteria_population_l4057_405700

/-- The number of seconds in 5 minutes -/
def totalTime : ℕ := 300

/-- The doubling time of the bacteria population in seconds -/
def doublingTime : ℕ := 30

/-- The number of bacteria after 5 minutes -/
def finalPopulation : ℕ := 1310720

/-- The number of doublings that occur in 5 minutes -/
def numberOfDoublings : ℕ := totalTime / doublingTime

theorem initial_bacteria_population :
  ∃ (initialPopulation : ℕ),
    initialPopulation * (2 ^ numberOfDoublings) = finalPopulation ∧
    initialPopulation = 1280 :=
by sorry

end NUMINAMATH_CALUDE_initial_bacteria_population_l4057_405700


namespace NUMINAMATH_CALUDE_problem_solution_l4057_405735

theorem problem_solution (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1) (h_x_z : x + 1 / z = 7) (h_y_x : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4057_405735


namespace NUMINAMATH_CALUDE_barbaras_candy_purchase_l4057_405785

/-- Theorem: Barbara's Candy Purchase
Given:
- initial_candies: The number of candies Barbara had initially
- final_candies: The number of candies Barbara has after buying more
- bought_candies: The number of candies Barbara bought

Prove that bought_candies = 18, given initial_candies = 9 and final_candies = 27
-/
theorem barbaras_candy_purchase 
  (initial_candies : ℕ) 
  (final_candies : ℕ) 
  (bought_candies : ℕ) 
  (h1 : initial_candies = 9)
  (h2 : final_candies = 27)
  (h3 : final_candies = initial_candies + bought_candies) :
  bought_candies = 18 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_candy_purchase_l4057_405785


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l4057_405718

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ) 
  (watermelon_salmon watermelon_chicken salmon_chicken : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_watermelon : watermelon = 20)
  (h_salmon : salmon = 70)
  (h_chicken : chicken = 10)
  (h_watermelon_salmon : watermelon_salmon = 10)
  (h_salmon_chicken : salmon_chicken = 5)
  (h_watermelon_chicken : watermelon_chicken = 3)
  (h_all_three : all_three = 2) :
  total - (watermelon + salmon + chicken - watermelon_salmon - watermelon_chicken - salmon_chicken + all_three) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l4057_405718


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l4057_405792

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem: A 30-inch by 24-inch pan can be divided into exactly 120 pieces of 3-inch by 2-inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 30, width := 24 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 120 := by
  sorry


end NUMINAMATH_CALUDE_brownie_pieces_count_l4057_405792


namespace NUMINAMATH_CALUDE_permutations_of_47722_l4057_405751

def digits : List ℕ := [4, 7, 7, 2, 2]

theorem permutations_of_47722 : Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_47722_l4057_405751


namespace NUMINAMATH_CALUDE_existence_implies_a_bound_l4057_405705

/-- Given a > 0, prove that if there exists x₀ ∈ (0, 1/2] such that f(x₀) > g(x₀), then a > -3 + √17 -/
theorem existence_implies_a_bound (a : ℝ) (h₁ : a > 0) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 (1/2) ∧ 
    a^2 * x₀^3 - 3*a * x₀^2 + 2 > -3*a * x₀ + 3) → 
  a > -3 + Real.sqrt 17 := by
sorry

/-- Definition of f(x) -/
def f (a x : ℝ) : ℝ := a^2 * x^3 - 3*a * x^2 + 2

/-- Definition of g(x) -/
def g (a x : ℝ) : ℝ := -3*a * x + 3

end NUMINAMATH_CALUDE_existence_implies_a_bound_l4057_405705


namespace NUMINAMATH_CALUDE_factory_output_increase_l4057_405739

theorem factory_output_increase (planned_output actual_output : ℝ) 
  (h1 : planned_output = 20)
  (h2 : actual_output = 24) : 
  (actual_output - planned_output) / planned_output = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_increase_l4057_405739


namespace NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l4057_405773

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space by its equation ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point P(2,3)
def P : Point2D := ⟨2, 3⟩

-- The two possible lines
def line1 : Line2D := ⟨3, -2, 0⟩
def line2 : Line2D := ⟨1, 1, -5⟩

-- The theorem to prove
theorem line_through_P_with_equal_intercepts :
  (pointOnLine P line1 ∧ equalIntercepts line1) ∨
  (pointOnLine P line2 ∧ equalIntercepts line2) := by
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l4057_405773


namespace NUMINAMATH_CALUDE_max_value_a_l4057_405790

theorem max_value_a (x y : ℝ) 
  (h1 : x - y ≤ 0) 
  (h2 : x + y - 5 ≥ 0) 
  (h3 : y - 3 ≤ 0) : 
  (∃ (a : ℝ), a = 25/13 ∧ 
    (∀ (b : ℝ), (∀ (x y : ℝ), 
      x - y ≤ 0 → x + y - 5 ≥ 0 → y - 3 ≤ 0 → 
      b * (x^2 + y^2) ≤ (x + y)^2) → 
    b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l4057_405790


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l4057_405767

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) : 
  (∀ x, p x = (7/8) * x^2 - (13/4) * x + 3) →
  p (-2) = 13 ∧ p 0 = 3 ∧ p 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l4057_405767


namespace NUMINAMATH_CALUDE_average_difference_l4057_405766

/-- Given that the average of a and b is 50, and the average of b and c is 70, prove that c - a = 40 -/
theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l4057_405766


namespace NUMINAMATH_CALUDE_formula_holds_for_given_pairs_l4057_405784

def formula (x : ℕ) : ℕ := x^2 + 4*x + 3

theorem formula_holds_for_given_pairs : 
  (formula 1 = 3) ∧ 
  (formula 2 = 8) ∧ 
  (formula 3 = 15) ∧ 
  (formula 4 = 24) ∧ 
  (formula 5 = 35) := by
  sorry

end NUMINAMATH_CALUDE_formula_holds_for_given_pairs_l4057_405784


namespace NUMINAMATH_CALUDE_flowers_per_pot_l4057_405736

theorem flowers_per_pot (num_gardens : ℕ) (pots_per_garden : ℕ) (total_flowers : ℕ) : 
  num_gardens = 10 →
  pots_per_garden = 544 →
  total_flowers = 174080 →
  total_flowers / (num_gardens * pots_per_garden) = 32 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_pot_l4057_405736


namespace NUMINAMATH_CALUDE_bottle_problem_l4057_405712

/-- Represents a bottle in the case -/
inductive Bottle
  | FirstPrize
  | SecondPrize
  | NoPrize

/-- Represents the case of bottles -/
def Case : Finset Bottle := sorry

/-- The number of bottles in the case -/
def caseSize : ℕ := 6

/-- The number of bottles with prizes -/
def prizeBottles : ℕ := 2

/-- The number of bottles without prizes -/
def noPrizeBottles : ℕ := 4

/-- Person A's selection of bottles -/
def Selection : Finset Bottle := sorry

/-- The number of bottles selected -/
def selectionSize : ℕ := 2

/-- Event A: A did not win a prize -/
def EventA : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ ∀ b ∈ s, b = Bottle.NoPrize}

/-- Event B: A won the first prize -/
def EventB : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ Bottle.FirstPrize ∈ s}

/-- Event C: A won a prize -/
def EventC : Set (Finset Bottle) :=
  {s | s ⊆ Case ∧ s.card = selectionSize ∧ (Bottle.FirstPrize ∈ s ∨ Bottle.SecondPrize ∈ s)}

/-- The probability measure on the sample space -/
noncomputable def P : Set (Finset Bottle) → ℝ := sorry

theorem bottle_problem :
  (EventA ∩ EventC = ∅) ∧ (P (EventB ∪ EventC) = P EventC) := by sorry

end NUMINAMATH_CALUDE_bottle_problem_l4057_405712


namespace NUMINAMATH_CALUDE_test_score_ranges_l4057_405733

/-- Given three ranges of test scores, prove that R1 is 30 -/
theorem test_score_ranges (R1 R2 R3 : ℕ) : 
  R2 = 26 → 
  R3 = 32 → 
  (min R1 (min R2 R3) = 30) → 
  R1 = 30 := by
sorry

end NUMINAMATH_CALUDE_test_score_ranges_l4057_405733


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_square_roots_solution_l4057_405710

theorem quadratic_equation_solution (x : ℝ) :
  25 * x^2 - 36 = 0 → x = 6/5 ∨ x = -6/5 := by sorry

theorem square_roots_solution (x a : ℝ) :
  a > 0 ∧ (x + 2)^2 = a ∧ (3*x - 10)^2 = a → x = 2 ∧ a = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_square_roots_solution_l4057_405710


namespace NUMINAMATH_CALUDE_symmetric_abs_sum_l4057_405726

/-- A function f is symmetric about a point c if f(c+x) = f(c-x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetric_abs_sum (a : ℝ) :
  SymmetricAbout (fun x ↦ |x + 1| + |x - a|) 1 → a = 3 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_abs_sum_l4057_405726


namespace NUMINAMATH_CALUDE_new_crust_flour_amount_l4057_405729

/-- The amount of flour per new pie crust when changing the recipe -/
def flour_per_new_crust (original_crusts : ℕ) (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) : ℚ :=
  (original_crusts : ℚ) * original_flour_per_crust / (new_crusts : ℚ)

/-- Theorem stating that the amount of flour per new pie crust is 1/5 cup -/
theorem new_crust_flour_amount : 
  flour_per_new_crust 40 (1/8) 25 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_new_crust_flour_amount_l4057_405729


namespace NUMINAMATH_CALUDE_window_washing_time_l4057_405761

/-- The time it takes your friend to wash a window (in minutes) -/
def friend_time : ℝ := 3

/-- The total time it takes both of you to wash 25 windows (in minutes) -/
def total_time : ℝ := 30

/-- The number of windows you wash together -/
def num_windows : ℝ := 25

/-- Your time to wash a window (in minutes) -/
def your_time : ℝ := 2

theorem window_washing_time :
  (1 / friend_time + 1 / your_time) * total_time = num_windows :=
sorry

end NUMINAMATH_CALUDE_window_washing_time_l4057_405761


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l4057_405798

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem lcm_hcf_problem (A B : ℕ) (h1 : Nat.lcm A B = 7700) (h2 : Nat.gcd A B = 11) (h3 : B = 275) :
  A = 308 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l4057_405798


namespace NUMINAMATH_CALUDE_special_triples_count_l4057_405742

/-- Represents a graph with a specific number of vertices and edges per vertex -/
structure Graph where
  numVertices : ℕ
  edgesPerVertex : ℕ

/-- Calculates the number of triples in a graph where each pair of vertices is either all connected or all disconnected -/
def countSpecialTriples (g : Graph) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem special_triples_count (g : Graph) (h1 : g.numVertices = 30) (h2 : g.edgesPerVertex = 6) :
  countSpecialTriples g = 1990 := by
  sorry

end NUMINAMATH_CALUDE_special_triples_count_l4057_405742


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l4057_405791

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℕ
  side : ℕ

/-- The theorem statement -/
theorem similar_triangles_side_length 
  (small large : Triangle)
  (h_diff : large.area - small.area = 32)
  (h_ratio : ∃ k : ℕ, large.area = k^2 * small.area)
  (h_small_side : small.side = 4) :
  large.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l4057_405791


namespace NUMINAMATH_CALUDE_salt_production_theorem_l4057_405713

/-- Calculates the average daily salt production for a year given the initial production and monthly increase. -/
def averageDailyProduction (initialProduction : ℕ) (monthlyIncrease : ℕ) : ℚ :=
  let totalProduction := initialProduction + (initialProduction + monthlyIncrease + initialProduction + monthlyIncrease * 11) * 11 / 2
  totalProduction / 365

/-- Theorem stating that the average daily production is approximately 83.84 tonnes. -/
theorem salt_production_theorem (initialProduction monthlyIncrease : ℕ) 
  (h1 : initialProduction = 2000)
  (h2 : monthlyIncrease = 100) :
  ∃ ε > 0, |averageDailyProduction initialProduction monthlyIncrease - 83.84| < ε :=
sorry

end NUMINAMATH_CALUDE_salt_production_theorem_l4057_405713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4057_405797

theorem arithmetic_sequence_third_term (a x : ℝ) : 
  a + (a + 2*x) = 6 → a + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l4057_405797


namespace NUMINAMATH_CALUDE_equal_group_formation_l4057_405714

-- Define the total number of people
def total_people : ℕ := 20

-- Define the number of boys
def num_boys : ℕ := 10

-- Define the number of girls
def num_girls : ℕ := 10

-- Define the size of the group to be formed
def group_size : ℕ := 10

-- Theorem statement
theorem equal_group_formation :
  Nat.choose total_people group_size = 184756 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_group_formation_l4057_405714


namespace NUMINAMATH_CALUDE_min_value_of_function_l4057_405755

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a^5 + a^5 - 2) * (1/b^5 + b^5 - 2) ≥ 31^4 / 32^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4057_405755


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l4057_405747

def complex (a b : ℝ) : ℂ := ⟨a, b⟩

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_square (x : ℝ) :
  is_pure_imaginary ((complex x 1)^2) → x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l4057_405747


namespace NUMINAMATH_CALUDE_lego_storage_time_l4057_405770

/-- The time needed to store all Lego pieces -/
def storage_time (total_pieces : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  (total_pieces - 1) / net_increase_per_minute + 1

/-- Theorem: It takes 43 minutes to store 45 Lego pieces with a net increase of 1 piece per minute -/
theorem lego_storage_time :
  storage_time 45 1 = 43 := by
  sorry

end NUMINAMATH_CALUDE_lego_storage_time_l4057_405770


namespace NUMINAMATH_CALUDE_archibald_win_percentage_l4057_405781

/-- Given that Archibald has won 12 games and his brother has won 18 games,
    prove that the percentage of games Archibald has won is 40%. -/
theorem archibald_win_percentage 
  (archibald_wins : ℕ) 
  (brother_wins : ℕ) 
  (h1 : archibald_wins = 12)
  (h2 : brother_wins = 18) :
  (archibald_wins : ℚ) / (archibald_wins + brother_wins) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_archibald_win_percentage_l4057_405781


namespace NUMINAMATH_CALUDE_square_root_equation_implies_y_minus_x_equals_two_l4057_405743

theorem square_root_equation_implies_y_minus_x_equals_two (x y : ℝ) :
  Real.sqrt (x + 1) - Real.sqrt (-1 - x) = (x + y)^2 → y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_implies_y_minus_x_equals_two_l4057_405743


namespace NUMINAMATH_CALUDE_cylinder_volume_l4057_405763

/-- Given a cylinder whose lateral surface unfolds into a rectangle with length 2a and width a, 
    its volume is either a³/π or a³/(2π) -/
theorem cylinder_volume (a : ℝ) (h : a > 0) :
  ∃ (V : ℝ), (V = a^3 / π ∨ V = a^3 / (2*π)) ∧
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  ((2*π*r = 2*a ∧ h = a) ∨ (2*π*r = a ∧ h = 2*a)) ∧
  V = π * r^2 * h :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_l4057_405763


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l4057_405732

-- Define the parallelogram properties
def base : ℝ := 20
def total_length : ℝ := 26
def slant_height : ℝ := 7

-- Define the area function for a parallelogram
def parallelogram_area (b h : ℝ) : ℝ := b * h

-- Theorem statement
theorem parallelogram_area_calculation :
  ∃ (height : ℝ), 
    height^2 + (total_length - base)^2 = slant_height^2 ∧
    parallelogram_area base height = 20 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l4057_405732


namespace NUMINAMATH_CALUDE_male_puppies_count_l4057_405745

/-- Proves that the number of male puppies is 10 given the specified conditions -/
theorem male_puppies_count (total_puppies : ℕ) (female_puppies : ℕ) (ratio : ℚ) :
  total_puppies = 12 →
  female_puppies = 2 →
  ratio = 1/5 →
  total_puppies = female_puppies + (female_puppies / ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_male_puppies_count_l4057_405745


namespace NUMINAMATH_CALUDE_little_john_remaining_money_l4057_405768

/-- Calculates the remaining money after Little John's expenditures -/
def remaining_money (initial_amount spent_on_sweets toy_cost friend_gift number_of_friends : ℚ) : ℚ :=
  initial_amount - (spent_on_sweets + toy_cost + friend_gift * number_of_friends)

/-- Theorem: Given Little John's initial amount and expenditures, the remaining money is $11.55 -/
theorem little_john_remaining_money :
  remaining_money 20.10 1.05 2.50 1.00 5 = 11.55 := by
  sorry

#eval remaining_money 20.10 1.05 2.50 1.00 5

end NUMINAMATH_CALUDE_little_john_remaining_money_l4057_405768


namespace NUMINAMATH_CALUDE_man_speed_man_speed_is_6_l4057_405701

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5/18)
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * (18/5)
  man_speed_kmph

/-- The speed of the man is 6 kmph --/
theorem man_speed_is_6 :
  man_speed 220 60 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_is_6_l4057_405701


namespace NUMINAMATH_CALUDE_sin_10_50_70_product_l4057_405775

theorem sin_10_50_70_product : Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_50_70_product_l4057_405775


namespace NUMINAMATH_CALUDE_age_difference_l4057_405746

theorem age_difference (tyson_age frederick_age julian_age kyle_age : ℕ) : 
  tyson_age = 20 →
  frederick_age = 2 * tyson_age →
  julian_age = frederick_age - 20 →
  kyle_age = 25 →
  kyle_age - julian_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l4057_405746


namespace NUMINAMATH_CALUDE_max_basketballs_l4057_405774

/-- The cost of footballs and basketballs -/
structure BallCosts where
  football : ℕ
  basketball : ℕ

/-- The problem setup -/
structure ProblemSetup where
  costs : BallCosts
  total_balls : ℕ
  max_cost : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (setup : ProblemSetup) : Prop :=
  3 * setup.costs.football + 2 * setup.costs.basketball = 310 ∧
  2 * setup.costs.football + 5 * setup.costs.basketball = 500 ∧
  setup.total_balls = 96 ∧
  setup.max_cost = 5800

/-- The theorem to prove -/
theorem max_basketballs (setup : ProblemSetup) 
  (h : satisfies_conditions setup) : 
  ∃ (x : ℕ), x ≤ setup.total_balls ∧ 
    x * setup.costs.basketball + (setup.total_balls - x) * setup.costs.football ≤ setup.max_cost ∧
    ∀ (y : ℕ), y > x → 
      y * setup.costs.basketball + (setup.total_balls - y) * setup.costs.football > setup.max_cost :=
by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_l4057_405774


namespace NUMINAMATH_CALUDE_fruit_store_solution_l4057_405787

/-- Represents the purchase quantities and costs of two types of fruits -/
structure FruitPurchase where
  quantityA : ℕ  -- Quantity of fruit A in kg
  quantityB : ℕ  -- Quantity of fruit B in kg
  totalCost : ℕ  -- Total cost in yuan

/-- The fruit store problem -/
def fruitStoreProblem (purchase1 purchase2 : FruitPurchase) : Prop :=
  ∃ (priceA priceB : ℕ),
    -- Conditions from the first purchase
    purchase1.quantityA * priceA + purchase1.quantityB * priceB = purchase1.totalCost ∧
    -- Conditions from the second purchase
    purchase2.quantityA * priceA + purchase2.quantityB * priceB = purchase2.totalCost ∧
    -- Unique solution condition
    ∀ (x y : ℕ),
      (purchase1.quantityA * x + purchase1.quantityB * y = purchase1.totalCost ∧
       purchase2.quantityA * x + purchase2.quantityB * y = purchase2.totalCost) →
      x = priceA ∧ y = priceB

/-- Theorem stating the solution to the fruit store problem -/
theorem fruit_store_solution :
  fruitStoreProblem
    { quantityA := 60, quantityB := 40, totalCost := 1520 }
    { quantityA := 30, quantityB := 50, totalCost := 1360 } →
  ∃ (priceA priceB : ℕ), priceA = 12 ∧ priceB = 20 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_solution_l4057_405787


namespace NUMINAMATH_CALUDE_sfl_entrances_l4057_405762

/-- Given that there are 283 people waiting at each entrance and 1415 people in total,
    prove that the number of entrances is 5. -/
theorem sfl_entrances (people_per_entrance : ℕ) (total_people : ℕ) 
  (h1 : people_per_entrance = 283) 
  (h2 : total_people = 1415) :
  total_people / people_per_entrance = 5 := by
  sorry

end NUMINAMATH_CALUDE_sfl_entrances_l4057_405762


namespace NUMINAMATH_CALUDE_brooke_homework_time_l4057_405721

/-- Calculates the total time Brooke spends on homework, including breaks -/
def total_homework_time (math_problems : ℕ) (social_studies_problems : ℕ) (science_problems : ℕ)
  (math_time_per_problem : ℚ) (social_studies_time_per_problem : ℚ) (science_time_per_problem : ℚ)
  (math_break : ℕ) (social_studies_break : ℕ) (science_break : ℕ) : ℚ :=
  let math_time := math_problems * math_time_per_problem
  let social_studies_time := social_studies_problems * social_studies_time_per_problem / 60
  let science_time := science_problems * science_time_per_problem
  let total_problem_time := math_time + social_studies_time + science_time
  let total_break_time := math_break + social_studies_break + science_break
  total_problem_time + total_break_time

theorem brooke_homework_time :
  total_homework_time 15 6 10 2 (1/2) (3/2) 5 10 15 = 78 := by
  sorry

end NUMINAMATH_CALUDE_brooke_homework_time_l4057_405721


namespace NUMINAMATH_CALUDE_unit_circle_complex_bound_l4057_405752

theorem unit_circle_complex_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (zmin zmax : ℂ),
    Complex.abs (z^3 - 3*z - 2) ≤ Real.sqrt 27 ∧
    Complex.abs (zmax^3 - 3*zmax - 2) = Real.sqrt 27 ∧
    Complex.abs (zmin^3 - 3*zmin - 2) = 0 ∧
    Complex.abs zmax = 1 ∧
    Complex.abs zmin = 1 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_complex_bound_l4057_405752


namespace NUMINAMATH_CALUDE_power_function_through_point_l4057_405708

theorem power_function_through_point (k m : ℝ) : 
  k * (2 : ℝ)^m = 1/4 → m * k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4057_405708


namespace NUMINAMATH_CALUDE_exists_polynomial_for_E_l4057_405740

/-- Definition of E(m) as described in the problem -/
def E (m : ℕ) : ℕ :=
  (Finset.univ.filter (fun s : Finset (Fin 6) => s.card = 6)).card

/-- The main theorem to be proved -/
theorem exists_polynomial_for_E :
  ∃ (c₄ c₃ c₂ c₁ c₀ : ℚ),
    ∀ (m : ℕ), m ≥ 6 → m % 2 = 0 →
      E m = c₄ * m^4 + c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀ := by
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_for_E_l4057_405740


namespace NUMINAMATH_CALUDE_sum_of_abc_l4057_405715

theorem sum_of_abc (a b c : ℝ) 
  (h1 : a^2*b + a^2*c + b^2*a + b^2*c + c^2*a + c^2*b + 3*a*b*c = 30)
  (h2 : a^2 + b^2 + c^2 = 13) : 
  a + b + c = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_abc_l4057_405715


namespace NUMINAMATH_CALUDE_log_not_always_decreasing_l4057_405709

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ∃ (a : ℝ), a > 1 ∧ ∀ (x y : ℝ), x > y → x > 0 → y > 0 → log a x > log a y :=
sorry

end NUMINAMATH_CALUDE_log_not_always_decreasing_l4057_405709


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_666_and_8_mixed_number_representation_l4057_405756

def repeating_decimal_666 : ℚ := 2/3

theorem product_of_repeating_decimal_666_and_8 :
  repeating_decimal_666 * 8 = 16/3 :=
sorry

theorem mixed_number_representation :
  16/3 = 5 + 1/3 :=
sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_666_and_8_mixed_number_representation_l4057_405756


namespace NUMINAMATH_CALUDE_min_distance_is_3420_div_181_l4057_405744

/-- Triangle ABC with right angle at B, side lengths, and intersecting circles --/
structure RightTriangleWithCircles where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angle condition
  right_angle : ab^2 + bc^2 = ac^2
  -- Side length values
  ab_eq : ab = 19
  bc_eq : bc = 180
  ac_eq : ac = 181
  -- Midpoints
  m : ℝ × ℝ  -- midpoint of AB
  n : ℝ × ℝ  -- midpoint of BC
  -- Intersection points
  d : ℝ × ℝ
  e : ℝ × ℝ
  p : ℝ × ℝ
  -- Conditions for D and E
  d_on_circle_m : (d.1 - m.1)^2 + (d.2 - m.2)^2 = (ac/2)^2
  d_on_circle_n : (d.1 - n.1)^2 + (d.2 - n.2)^2 = (ac/2)^2
  e_on_circle_m : (e.1 - m.1)^2 + (e.2 - m.2)^2 = (ac/2)^2
  e_on_circle_n : (e.1 - n.1)^2 + (e.2 - n.2)^2 = (ac/2)^2
  -- P is on AC
  p_on_ac : p.2 = 0
  -- DE intersects AC at P
  p_on_de : ∃ (t : ℝ), p = (1 - t) • d + t • e

/-- The minimum of DP and EP is 3420/181 --/
theorem min_distance_is_3420_div_181 (triangle : RightTriangleWithCircles) :
  min ((triangle.d.1 - triangle.p.1)^2 + (triangle.d.2 - triangle.p.2)^2)
      ((triangle.e.1 - triangle.p.1)^2 + (triangle.e.2 - triangle.p.2)^2) = (3420/181)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_is_3420_div_181_l4057_405744


namespace NUMINAMATH_CALUDE_circle_center_sum_l4057_405765

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4057_405765


namespace NUMINAMATH_CALUDE_extreme_value_conditions_l4057_405795

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_conditions (a b : ℝ) : 
  f a b (-1) = 8 ∧ f_derivative a b (-1) = 0 → a = 2 ∧ b = -7 := by sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_l4057_405795


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l4057_405754

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Counts distinct equilateral triangles with at least two vertices from the polygon -/
def count_equilateral_triangles (p : RegularPolygon 11) : ℕ :=
  sorry

/-- The main theorem stating the count of distinct equilateral triangles -/
theorem equilateral_triangle_count (p : RegularPolygon 11) :
  count_equilateral_triangles p = 92 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l4057_405754


namespace NUMINAMATH_CALUDE_factorize_expression_1_l4057_405728

theorem factorize_expression_1 (x : ℝ) :
  (x^2 - 1 + x) * (x^2 - 1 + 3*x) + x^2 = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 := by
sorry

end NUMINAMATH_CALUDE_factorize_expression_1_l4057_405728


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l4057_405719

theorem complex_sum_of_parts (a b : ℝ) (h : (Complex.mk a b) = Complex.mk 1 (-1)) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l4057_405719


namespace NUMINAMATH_CALUDE_root_property_l4057_405757

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x - a

-- State the theorem
theorem root_property (a : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 2)
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : f a x₃ = 0)
  (h₆ : x₁ < x₂) (h₇ : x₂ < x₃) :
  x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l4057_405757


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l4057_405730

theorem parabola_point_to_directrix_distance 
  (C : ℝ → ℝ → Prop) 
  (p : ℝ) 
  (A : ℝ × ℝ) :
  (∀ x y, C x y ↔ y^2 = 2*p*x) →  -- Definition of parabola C
  C A.1 A.2 →  -- A lies on C
  A = (1, Real.sqrt 5) →  -- Coordinates of A
  (A.1 + p/2) = 9/4 :=  -- Distance formula to directrix
by sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l4057_405730


namespace NUMINAMATH_CALUDE_coal_division_l4057_405786

/-- Given 3 tons of coal divided equally into 5 parts, prove the fraction and amount of each part -/
theorem coal_division (total_coal : ℝ) (num_parts : ℕ) 
  (h1 : total_coal = 3)
  (h2 : num_parts = 5) :
  (1 : ℝ) / num_parts = 1 / 5 ∧ 
  total_coal / num_parts = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_coal_division_l4057_405786


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speed_problem_l4057_405704

/-- The problem setup and solution for the pedestrian and cyclist speed problem -/
theorem pedestrian_cyclist_speed_problem :
  let distance : ℝ := 40 -- km
  let pedestrian_start_time : ℝ := 0 -- 4:00 AM
  let first_cyclist_start_time : ℝ := 3 + 1/3 -- 7:20 AM
  let second_cyclist_start_time : ℝ := 4.5 -- 8:30 AM
  let meetup_distance : ℝ := distance / 2
  let second_meetup_time_diff : ℝ := 1 -- hour

  ∃ (pedestrian_speed cyclist_speed : ℝ),
    pedestrian_speed > 0 ∧
    cyclist_speed > 0 ∧
    pedestrian_speed < cyclist_speed ∧
    -- First cyclist catches up with pedestrian at midpoint
    meetup_distance = pedestrian_speed * (first_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed)) ∧
    -- Second cyclist meets pedestrian one hour after first meetup
    distance = pedestrian_speed * (second_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff) + 
      cyclist_speed * ((distance - meetup_distance) / cyclist_speed - 
      ((meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff)) ∧
    pedestrian_speed = 5 ∧
    cyclist_speed = 30
  := by sorry

end NUMINAMATH_CALUDE_pedestrian_cyclist_speed_problem_l4057_405704


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l4057_405724

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose three diagonals that intersect at a single point -/
def num_intersecting_diagonals (d : RegularDecagon) : ℕ := 840

/-- The total number of ways to choose three diagonals -/
def total_diagonal_choices (d : RegularDecagon) : ℕ := 6545

/-- The probability that three randomly chosen diagonals in a regular decagon
    intersect at a single point inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_intersecting_diagonals d : ℚ) / (total_diagonal_choices d : ℚ) = 840 / 6545 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l4057_405724


namespace NUMINAMATH_CALUDE_f_min_value_l4057_405789

/-- The function f(x) = |3-x| + |x-7| -/
def f (x : ℝ) := |3 - x| + |x - 7|

/-- The minimum value of f(x) is 4 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 4 ∧ ∃ y : ℝ, f y = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l4057_405789


namespace NUMINAMATH_CALUDE_function_max_implies_a_range_l4057_405722

/-- Given a function f(x) = (ax^2)/2 - (1+2a)x + 2ln(x) where a > 0,
    if f(x) has a maximum value in the interval (1/2, 1),
    then 1 < a < 2. -/
theorem function_max_implies_a_range (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = (a * x^2) / 2 - (1 + 2*a) * x + 2 * Real.log x) →
  (∃ x₀ ∈ Set.Ioo (1/2) 1, ∀ x ∈ Set.Ioo (1/2) 1, f x ≤ f x₀) →
  1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_function_max_implies_a_range_l4057_405722


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_percent_l4057_405760

def total_work_hours : ℕ := 8
def first_meeting_duration : ℕ := 30
def minutes_per_hour : ℕ := 60

def total_work_minutes : ℕ := total_work_hours * minutes_per_hour
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem meeting_percentage_is_37_5_percent :
  (total_meeting_duration : ℚ) / (total_work_minutes : ℚ) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_percent_l4057_405760


namespace NUMINAMATH_CALUDE_upper_bound_of_expression_l4057_405706

theorem upper_bound_of_expression (n : ℤ) (U : ℤ) : 
  (∃ (S : Finset ℤ), 
    (∀ m ∈ S, (4 * m + 7 > 1 ∧ 4 * m + 7 < U)) ∧ 
    S.card = 15 ∧
    (∀ m : ℤ, 4 * m + 7 > 1 ∧ 4 * m + 7 < U → m ∈ S)) →
  U ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_upper_bound_of_expression_l4057_405706


namespace NUMINAMATH_CALUDE_negation_equivalence_l4057_405737

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x < 2 ∧ x^2 - 2*x < 0)) ↔ (∀ x : ℝ, x < 2 → x^2 - 2*x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4057_405737


namespace NUMINAMATH_CALUDE_fraction_value_l4057_405764

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l4057_405764


namespace NUMINAMATH_CALUDE_max_product_difference_l4057_405741

theorem max_product_difference (x y : ℕ) : 
  x > 0 → y > 0 → x + 2 * y = 2008 → (∀ a b : ℕ, a > 0 → b > 0 → a + 2 * b = 2008 → x * y ≥ a * b) → x - y = 502 := by
sorry

end NUMINAMATH_CALUDE_max_product_difference_l4057_405741


namespace NUMINAMATH_CALUDE_equation_solutions_l4057_405793

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- State the theorem
theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 31 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4057_405793


namespace NUMINAMATH_CALUDE_iggy_ran_four_miles_on_tuesday_l4057_405748

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running data --/
structure RunningData where
  miles : RunDay → ℕ
  pace : ℕ  -- minutes per mile
  totalTime : ℕ  -- total running time in minutes

/-- Theorem stating that Iggy ran 4 miles on Tuesday --/
theorem iggy_ran_four_miles_on_tuesday (data : RunningData) : data.miles RunDay.Tuesday = 4 :=
  by
  have h1 : data.miles RunDay.Monday = 3 := by sorry
  have h2 : data.miles RunDay.Wednesday = 6 := by sorry
  have h3 : data.miles RunDay.Thursday = 8 := by sorry
  have h4 : data.miles RunDay.Friday = 3 := by sorry
  have h5 : data.pace = 10 := by sorry
  have h6 : data.totalTime = 4 * 60 := by sorry
  
  sorry


end NUMINAMATH_CALUDE_iggy_ran_four_miles_on_tuesday_l4057_405748


namespace NUMINAMATH_CALUDE_company_sampling_methods_l4057_405777

/-- Enumeration of regions --/
inductive Region
| A
| B
| C
| D

/-- Enumeration of sampling methods --/
inductive SamplingMethod
| StratifiedSampling
| SimpleRandomSampling

/-- Structure representing the sales points distribution --/
structure SalesDistribution where
  total_points : ℕ
  region_points : Region → ℕ
  large_points_C : ℕ

/-- Structure representing an investigation --/
structure Investigation where
  sample_size : ℕ
  population_size : ℕ

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (dist : SalesDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given scenario --/
theorem company_sampling_methods 
  (dist : SalesDistribution)
  (inv1 inv2 : Investigation)
  (h1 : dist.total_points = 600)
  (h2 : dist.region_points Region.A = 150)
  (h3 : dist.region_points Region.B = 120)
  (h4 : dist.region_points Region.C = 180)
  (h5 : dist.region_points Region.D = 150)
  (h6 : dist.large_points_C = 20)
  (h7 : inv1.sample_size = 100)
  (h8 : inv1.population_size = 600)
  (h9 : inv2.sample_size = 7)
  (h10 : inv2.population_size = 20) :
  (appropriate_sampling_method dist inv1 = SamplingMethod.StratifiedSampling) ∧
  (appropriate_sampling_method dist inv2 = SamplingMethod.SimpleRandomSampling) :=
sorry

end NUMINAMATH_CALUDE_company_sampling_methods_l4057_405777
