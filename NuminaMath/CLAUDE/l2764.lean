import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2764_276481

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔ 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2764_276481


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l2764_276463

theorem sum_of_even_integers (a b c d : ℤ) 
  (h1 : Even a) (h2 : Even b) (h3 : Even c) (h4 : Even d)
  (eq1 : a - b + c = 8)
  (eq2 : b - c + d = 10)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 6) :
  a + b + c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l2764_276463


namespace NUMINAMATH_CALUDE_equal_color_distribution_l2764_276418

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of having exactly 4 black and 4 white balls -/
def prob_four_black_four_white : ℚ :=
  (choose n (n/2) : ℚ) * p^n

theorem equal_color_distribution :
  prob_four_black_four_white = 35/128 :=
sorry

end NUMINAMATH_CALUDE_equal_color_distribution_l2764_276418


namespace NUMINAMATH_CALUDE_garden_fencing_length_l2764_276426

theorem garden_fencing_length (garden_area : ℝ) (π_approx : ℝ) (extra_length : ℝ) : 
  garden_area = 616 → 
  π_approx = 22 / 7 → 
  extra_length = 5 → 
  2 * π_approx * Real.sqrt (garden_area / π_approx) + extra_length = 93 := by
sorry

end NUMINAMATH_CALUDE_garden_fencing_length_l2764_276426


namespace NUMINAMATH_CALUDE_inequalities_hold_l2764_276454

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2764_276454


namespace NUMINAMATH_CALUDE_bigger_part_of_division_l2764_276423

theorem bigger_part_of_division (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 60) (h4 : 10 * x + 22 * y = 780) : max x y = 45 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_of_division_l2764_276423


namespace NUMINAMATH_CALUDE_pass_in_later_rounds_l2764_276475

/-- Represents the probability of correctly answering each question -/
structure QuestionProbabilities where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the interview process -/
def Interview (probs : QuestionProbabilities) : Prop :=
  probs.A = 1/2 ∧ probs.B = 1/3 ∧ probs.C = 1/4

/-- The probability of passing the interview in the second or third round -/
def PassInLaterRounds (probs : QuestionProbabilities) : ℚ :=
  7/18

/-- Theorem stating the probability of passing in later rounds -/
theorem pass_in_later_rounds (probs : QuestionProbabilities) 
  (h : Interview probs) : 
  PassInLaterRounds probs = 7/18 := by
  sorry


end NUMINAMATH_CALUDE_pass_in_later_rounds_l2764_276475


namespace NUMINAMATH_CALUDE_inequality_implication_l2764_276457

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2764_276457


namespace NUMINAMATH_CALUDE_exam_score_below_mean_l2764_276420

/-- Given an exam with mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean 
  (mean : ℝ) 
  (score_above : ℝ) 
  (sd_above : ℝ) 
  (sd_below : ℝ) 
  (h1 : mean = 88.8)
  (h2 : score_above = 90)
  (h3 : sd_above = 3)
  (h4 : sd_below = 7)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 86 := by
sorry


end NUMINAMATH_CALUDE_exam_score_below_mean_l2764_276420


namespace NUMINAMATH_CALUDE_vector_collinearity_l2764_276428

theorem vector_collinearity (m n : ℝ) (h : n ≠ 0) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, 3]
  (∃ (k : ℝ), k ≠ 0 ∧ (fun i => m * a i - n * b i) = (fun i => k * (a i + 2 * b i))) →
  m / n = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2764_276428


namespace NUMINAMATH_CALUDE_stating_parallelogram_count_theorem_l2764_276404

/-- 
Given a triangle ABC where each side is divided into n equal parts and lines are drawn 
parallel to the sides through each division point, the function returns the total number 
of parallelograms formed in the resulting figure.
-/
def parallelogram_count (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- 
Theorem stating that the number of parallelograms in a triangle with sides divided into 
n equal parts and lines drawn parallel to sides through division points is 
3 * (n+2 choose 4).
-/
theorem parallelogram_count_theorem (n : ℕ) : 
  parallelogram_count n = 3 * Nat.choose (n + 2) 4 := by
  sorry

#eval parallelogram_count 5  -- Example evaluation

end NUMINAMATH_CALUDE_stating_parallelogram_count_theorem_l2764_276404


namespace NUMINAMATH_CALUDE_perfect_square_property_l2764_276411

theorem perfect_square_property (n : ℤ) : 
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) → 
  ∃ m : ℤ, (2 + 2 * Real.sqrt (1 + 12 * n^2))^2 = m^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2764_276411


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2764_276415

/-- The area of the shaded region in a geometric figure with the following properties:
    - A large square with side length 20 cm
    - Four quarter circles with radius 10 cm centered at the corners of the large square
    - A smaller square with side length 10 cm centered inside the larger square
    is equal to 100π - 100 cm². -/
theorem shaded_area_calculation (π : ℝ) : ℝ := by
  -- Define the side lengths and radius
  let large_square_side : ℝ := 20
  let small_square_side : ℝ := 10
  let quarter_circle_radius : ℝ := 10

  -- Define the areas
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2

  -- Calculate the shaded area
  let shaded_area : ℝ := quarter_circles_area - small_square_area

  -- Prove that the shaded area equals 100π - 100
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2764_276415


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2764_276476

theorem smallest_common_multiple_of_9_and_6 :
  ∃ n : ℕ+, (n : ℕ) % 9 = 0 ∧ (n : ℕ) % 6 = 0 ∧
  ∀ m : ℕ+, (m : ℕ) % 9 = 0 → (m : ℕ) % 6 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2764_276476


namespace NUMINAMATH_CALUDE_total_weight_is_7000_l2764_276427

/-- The weight of the truck in pounds -/
def truck_weight : ℝ := 4800

/-- The weight of the trailer in pounds -/
def trailer_weight : ℝ := 0.5 * truck_weight - 200

/-- The total weight of the truck and trailer in pounds -/
def total_weight : ℝ := truck_weight + trailer_weight

/-- Theorem stating that the total weight of the truck and trailer is 7000 pounds -/
theorem total_weight_is_7000 : total_weight = 7000 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_7000_l2764_276427


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2764_276467

/-- Represents the price reduction and resulting sales and profit changes for a toy product. -/
structure ToyPricing where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit after price reduction. -/
def dailyProfitAfterReduction (t : ToyPricing) : ℕ :=
  (t.initialProfit - t.priceReduction) * (t.initialSales + t.salesIncrease * t.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in the target daily profit. -/
theorem price_reduction_achieves_target_profit (t : ToyPricing) 
  (h1 : t.initialSales = 20)
  (h2 : t.initialProfit = 40)
  (h3 : t.salesIncrease = 2)
  (h4 : t.targetProfit = 1200)
  (h5 : t.priceReduction = 20) :
  dailyProfitAfterReduction t = t.targetProfit :=
by
  sorry

#eval dailyProfitAfterReduction { 
  initialSales := 20, 
  initialProfit := 40, 
  salesIncrease := 2, 
  priceReduction := 20, 
  targetProfit := 1200 
}

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l2764_276467


namespace NUMINAMATH_CALUDE_inverse_sum_product_l2764_276416

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_sum : 3*x + y/3 ≠ 0) : 
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = (x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l2764_276416


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2764_276403

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2764_276403


namespace NUMINAMATH_CALUDE_remainder_of_sum_l2764_276444

def start_num : ℕ := 11085

theorem remainder_of_sum (start : ℕ) (h : start = start_num) : 
  (2 * (List.sum (List.map (λ i => start + 2 * i) (List.range 8)))) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l2764_276444


namespace NUMINAMATH_CALUDE_sugar_servings_calculation_l2764_276408

/-- Calculates the number of servings in a container given the total amount and serving size -/
def number_of_servings (total_amount : ℚ) (serving_size : ℚ) : ℚ :=
  total_amount / serving_size

/-- Proves that a container with 35 2/3 cups of sugar contains 23 7/9 servings when each serving is 1 1/2 cups -/
theorem sugar_servings_calculation :
  let total_sugar : ℚ := 35 + 2/3
  let serving_size : ℚ := 1 + 1/2
  number_of_servings total_sugar serving_size = 23 + 7/9 := by
  sorry

#eval number_of_servings (35 + 2/3) (1 + 1/2)

end NUMINAMATH_CALUDE_sugar_servings_calculation_l2764_276408


namespace NUMINAMATH_CALUDE_vector_sum_necessary_not_sufficient_l2764_276439

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def form_triangle (a b c : V) : Prop := sorry

theorem vector_sum_necessary_not_sufficient (a b c : V) :
  (form_triangle a b c → a + b + c = 0) ∧
  ¬(a + b + c = 0 → form_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_necessary_not_sufficient_l2764_276439


namespace NUMINAMATH_CALUDE_nine_to_150_mod_50_l2764_276414

theorem nine_to_150_mod_50 : 9^150 % 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_150_mod_50_l2764_276414


namespace NUMINAMATH_CALUDE_simplify_expression_l2764_276453

theorem simplify_expression : 
  Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108) = 
  Real.sqrt 15 + 3 * Real.sqrt 5 + (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2764_276453


namespace NUMINAMATH_CALUDE_two_squares_share_vertices_l2764_276498

/-- A square in a plane. -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- An isosceles right triangle in a plane. -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a square shares two vertices with a triangle. -/
def SharesTwoVertices (s : Square) (t : IsoscelesRightTriangle) : Prop :=
  ∃ (i j : Fin 4) (v w : ℝ × ℝ), i ≠ j ∧
    s.vertices i = v ∧ s.vertices j = w ∧
    (v = t.A ∨ v = t.B ∨ v = t.C) ∧
    (w = t.A ∨ w = t.B ∨ w = t.C)

/-- The main theorem stating that there are exactly two squares sharing two vertices
    with an isosceles right triangle. -/
theorem two_squares_share_vertices (t : IsoscelesRightTriangle) :
  ∃! (n : ℕ), ∃ (squares : Fin n → Square),
    (∀ i, SharesTwoVertices (squares i) t) ∧
    (∀ s, SharesTwoVertices s t → ∃ i, s = squares i) ∧
    n = 2 :=
sorry

end NUMINAMATH_CALUDE_two_squares_share_vertices_l2764_276498


namespace NUMINAMATH_CALUDE_cubic_sum_identity_l2764_276484

theorem cubic_sum_identity 
  (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_identity_l2764_276484


namespace NUMINAMATH_CALUDE_miscalculation_correction_l2764_276407

theorem miscalculation_correction (x : ℝ) : 
  63 + x = 69 → 36 / x = 6 := by sorry

end NUMINAMATH_CALUDE_miscalculation_correction_l2764_276407


namespace NUMINAMATH_CALUDE_select_chess_team_l2764_276448

/-- The number of ways to select a team of 4 players from 10, where two are twins and both twins can't be on the team -/
def select_team (total_players : ℕ) (team_size : ℕ) (num_twins : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - num_twins) (team_size - num_twins)

/-- Theorem stating that selecting 4 players from 10, where two are twins and both twins can't be on the team, results in 182 ways -/
theorem select_chess_team : select_team 10 4 2 = 182 := by
  sorry

end NUMINAMATH_CALUDE_select_chess_team_l2764_276448


namespace NUMINAMATH_CALUDE_boat_downstream_time_l2764_276435

theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (current_rate : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 18) 
  (h2 : current_rate = 4) 
  (h3 : distance = 5.133333333333334) : 
  (distance / (boat_speed + current_rate)) * 60 = 14 := by
sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l2764_276435


namespace NUMINAMATH_CALUDE_special_function_value_l2764_276412

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 3) ≤ f x + 3) ∧
  (∀ x, f (x + 2) ≥ f x + 2) ∧
  (f 4 = 2008)

/-- The theorem to be proved -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) : f 2008 = 4012 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2764_276412


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2764_276413

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 4) : 
  x^4 + y^4 = 8432 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2764_276413


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2764_276424

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / a - y / b = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2764_276424


namespace NUMINAMATH_CALUDE_root_product_zero_l2764_276436

theorem root_product_zero (α β c : ℝ) : 
  (α^2 - 4*α + c = 0) → 
  (β^2 - 4*β + c = 0) → 
  ((-α)^2 + 4*(-α) - c = 0) → 
  α * β = 0 := by
sorry

end NUMINAMATH_CALUDE_root_product_zero_l2764_276436


namespace NUMINAMATH_CALUDE_flour_needed_l2764_276487

/-- Given a recipe requiring 12 cups of flour and 10 cups already added,
    prove that the additional cups of flour needed is 2. -/
theorem flour_needed (recipe_flour : ℕ) (added_flour : ℕ) : 
  recipe_flour = 12 → added_flour = 10 → recipe_flour - added_flour = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l2764_276487


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2764_276410

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8|

/-- The theorem stating that the sum of the maximum and minimum values of g(x) over [1, 10] is 2 -/
theorem sum_of_max_min_g :
  (⨆ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) + (⨅ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2764_276410


namespace NUMINAMATH_CALUDE_domain_equivalence_l2764_276437

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the domain of f(2x-1)
def domain_f_2x_minus_1 (f : ℝ → ℝ) : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem domain_equivalence (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0) →
  (∀ x, x ∈ domain_f_2x_minus_1 f ↔ f (2 * x - 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_domain_equivalence_l2764_276437


namespace NUMINAMATH_CALUDE_scooter_price_proof_l2764_276421

theorem scooter_price_proof (initial_price : ℝ) : 
  (∃ (total_cost selling_price : ℝ),
    total_cost = initial_price + 300 ∧
    selling_price = 1260 ∧
    selling_price = total_cost * 1.05) →
  initial_price = 900 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_proof_l2764_276421


namespace NUMINAMATH_CALUDE_intersection_sum_l2764_276474

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4
def g (x y : ℝ) : Prop := x + 5*y = 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    x₁ + x₂ + x₃ = 5 ∧
    y₁ + y₂ + y₃ = 2 :=
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2764_276474


namespace NUMINAMATH_CALUDE_okeydokey_investment_l2764_276440

/-- Represents the investment scenario for earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  artichokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Okeydokey invested -/
def okeydokey_apples (investment : EarthwormInvestment) : ℕ :=
  (investment.okeydokey_earthworms * (investment.artichokey_apples + investment.okeydokey_earthworms)) / 
  (investment.total_earthworms - investment.okeydokey_earthworms)

/-- Theorem stating that Okeydokey invested 5 apples -/
theorem okeydokey_investment (investment : EarthwormInvestment) 
  (h1 : investment.total_earthworms = 60)
  (h2 : investment.artichokey_apples = 7)
  (h3 : investment.okeydokey_earthworms = 25) : 
  okeydokey_apples investment = 5 := by
  sorry

#eval okeydokey_apples { total_earthworms := 60, artichokey_apples := 7, okeydokey_earthworms := 25 }

end NUMINAMATH_CALUDE_okeydokey_investment_l2764_276440


namespace NUMINAMATH_CALUDE_patio_layout_change_l2764_276443

theorem patio_layout_change (total_tiles : ℕ) (original_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 96 →
  original_rows = 8 →
  added_rows = 4 →
  (total_tiles / original_rows) - (total_tiles / (original_rows + added_rows)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_patio_layout_change_l2764_276443


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2764_276433

theorem largest_lcm_with_15 : 
  (Finset.image (fun x => Nat.lcm 15 x) {3, 5, 9, 12, 10, 15}).max = some 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2764_276433


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2764_276480

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ 
  (x : ℤ) ≡ 4 [ZMOD 7] ∧
  ∀ y : ℕ+, ((y : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ (y : ℤ) ≡ 4 [ZMOD 7]) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2764_276480


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l2764_276442

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x * (1 + p / 100) = x + x * (p / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l2764_276442


namespace NUMINAMATH_CALUDE_smallest_a_value_l2764_276465

theorem smallest_a_value (a b d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) :
  (∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x + d)) →
  ∃ k : ℤ, a = 17 - 2 * Real.pi * ↑k ∧ 
    ∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x + d)) → 
      17 - 2 * Real.pi * ↑k ≤ a' :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2764_276465


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_gasoline_tank_capacity_proof_l2764_276455

theorem gasoline_tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℝ := 5/6
    let final_fraction : ℝ := 1/3
    let used_amount : ℝ := 15
    initial_fraction * capacity - final_fraction * capacity = used_amount →
    capacity = 30

-- The proof goes here
theorem gasoline_tank_capacity_proof : gasoline_tank_capacity 30 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_gasoline_tank_capacity_proof_l2764_276455


namespace NUMINAMATH_CALUDE_scientific_notation_of_wetland_area_l2764_276466

/-- Proves that 29.47 thousand is equal to 2.947 × 10^4 in scientific notation -/
theorem scientific_notation_of_wetland_area :
  (29.47 * 1000 : ℝ) = 2.947 * (10 ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_wetland_area_l2764_276466


namespace NUMINAMATH_CALUDE_total_books_eq_read_plus_unread_l2764_276434

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 20

/-- The number of books yet to be read -/
def unread_books : ℕ := 5

/-- The number of books already read -/
def read_books : ℕ := 15

/-- Theorem stating that the total number of books is the sum of read and unread books -/
theorem total_books_eq_read_plus_unread : 
  total_books = read_books + unread_books := by
  sorry

end NUMINAMATH_CALUDE_total_books_eq_read_plus_unread_l2764_276434


namespace NUMINAMATH_CALUDE_min_value_theorem_l2764_276473

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  9 ≤ 4*a + b ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ 4*a₀ + b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2764_276473


namespace NUMINAMATH_CALUDE_number_equation_l2764_276406

theorem number_equation : ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2764_276406


namespace NUMINAMATH_CALUDE_three_zeros_range_of_a_l2764_276485

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a^2*x - 4*a

-- State the theorem
theorem three_zeros_range_of_a (a : ℝ) :
  a > 0 ∧ (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_range_of_a_l2764_276485


namespace NUMINAMATH_CALUDE_angle_sum_proof_l2764_276446

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : (1 - Real.tan α) * (1 - Real.tan β) = 2) : α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l2764_276446


namespace NUMINAMATH_CALUDE_h2so4_equals_khso4_l2764_276494

/-- Represents the balanced chemical equation for the reaction between KOH and H2SO4 to form KHSO4 -/
structure ChemicalReaction where
  koh : ℝ
  h2so4 : ℝ
  khso4 : ℝ

/-- The theorem states that the number of moles of H2SO4 needed is equal to the number of moles of KHSO4 formed,
    given that the number of moles of KOH initially present is equal to the number of moles of KHSO4 formed -/
theorem h2so4_equals_khso4 (reaction : ChemicalReaction) 
    (h : reaction.koh = reaction.khso4) : reaction.h2so4 = reaction.khso4 := by
  sorry

#check h2so4_equals_khso4

end NUMINAMATH_CALUDE_h2so4_equals_khso4_l2764_276494


namespace NUMINAMATH_CALUDE_correct_factorization_l2764_276468

theorem correct_factorization (x : ℝ) : 2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2764_276468


namespace NUMINAMATH_CALUDE_angle_A_value_min_value_expression_l2764_276402

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real -- circumradius

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  2 * t.R - t.a = (t.a * (t.b^2 + t.c^2 - t.a^2)) / (t.a^2 + t.c^2 - t.b^2)

-- Theorem 1
theorem angle_A_value (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) 
  (h3 : t.B = π/6) : 
  t.A = π/6 := by sorry

-- Theorem 2
theorem min_value_expression (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.A ≠ π/2) : 
  ∃ (min : Real), (∀ (t' : Triangle), triangle_condition t' → t'.A ≠ π/2 → 
    (2 * t'.a^2 - t'.c^2) / t'.b^2 ≥ min) ∧ 
  min = 4 * Real.sqrt 2 - 7 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_min_value_expression_l2764_276402


namespace NUMINAMATH_CALUDE_medical_team_selection_l2764_276469

theorem medical_team_selection (m n : ℕ) (hm : m = 6) (hn : n = 5) :
  (m.choose 2) * (n.choose 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l2764_276469


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l2764_276458

/-- Theorem: Diagonal of a rectangular prism with specific dimensions --/
theorem rectangular_prism_diagonal (h : ℝ) (l : ℝ) (w : ℝ) : 
  h = 12 → l = 2 * h → w = l / 2 → 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

/-- Corollary: Specific case with h = 12 --/
theorem rectangular_prism_diagonal_h12 : 
  ∃ (h l w : ℝ), h = 12 ∧ l = 2 * h ∧ w = l / 2 ∧ 
  Real.sqrt (l^2 + w^2 + h^2) = 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_rectangular_prism_diagonal_h12_l2764_276458


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l2764_276409

theorem continued_fraction_evaluation :
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l2764_276409


namespace NUMINAMATH_CALUDE_ratio_of_squares_nonnegative_l2764_276493

theorem ratio_of_squares_nonnegative (x : ℝ) (h : x ≠ 5) : (x^2) / ((x - 5)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_nonnegative_l2764_276493


namespace NUMINAMATH_CALUDE_figure_can_form_square_l2764_276459

/-- Represents a figure drawn on a grid -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- Represents a cut of the figure -/
structure Cut where
  piece1 : Set (ℤ × ℤ)
  piece2 : Set (ℤ × ℤ)
  piece3 : Set (ℤ × ℤ)

/-- Checks if a set of cells forms a square -/
def isSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (x y w : ℤ), ∀ (i j : ℤ), (i, j) ∈ s ↔ x ≤ i ∧ i < x + w ∧ y ≤ j ∧ j < y + w

/-- Theorem stating that the figure can be cut into three parts and reassembled into a square -/
theorem figure_can_form_square (f : GridFigure) :
  ∃ (c : Cut), c.piece1 ∪ c.piece2 ∪ c.piece3 = f.cells ∧
               isSquare (c.piece1 ∪ c.piece2 ∪ c.piece3) :=
sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l2764_276459


namespace NUMINAMATH_CALUDE_min_stamps_l2764_276477

def stamp_problem (n_010 n_020 n_050 n_200 : ℕ) (total : ℚ) : Prop :=
  n_010 ≥ 2 ∧
  n_020 ≥ 5 ∧
  n_050 ≥ 3 ∧
  n_200 ≥ 1 ∧
  total = 10 ∧
  0.1 * n_010 + 0.2 * n_020 + 0.5 * n_050 + 2 * n_200 = total

theorem min_stamps :
  ∃ (n_010 n_020 n_050 n_200 : ℕ),
    stamp_problem n_010 n_020 n_050 n_200 10 ∧
    (∀ (m_010 m_020 m_050 m_200 : ℕ),
      stamp_problem m_010 m_020 m_050 m_200 10 →
      n_010 + n_020 + n_050 + n_200 ≤ m_010 + m_020 + m_050 + m_200) ∧
    n_010 + n_020 + n_050 + n_200 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_min_stamps_l2764_276477


namespace NUMINAMATH_CALUDE_geometric_roots_product_l2764_276452

/-- 
Given an equation (x² - mx - 8)(x² - nx - 8) = 0 where m and n are real numbers,
if its four roots form a geometric sequence with the first term being 1,
then the product of m and n is -14.
-/
theorem geometric_roots_product (m n : ℝ) : 
  (∃ a r : ℝ, r ≠ 0 ∧ a = 1 ∧
    (∀ x : ℝ, (x^2 - m*x - 8 = 0 ∨ x^2 - n*x - 8 = 0) ↔ 
      (x = a ∨ x = a*r ∨ x = a*r^2 ∨ x = a*r^3))) →
  m * n = -14 := by
sorry

end NUMINAMATH_CALUDE_geometric_roots_product_l2764_276452


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2764_276464

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -3) :
  let a := geometric_sequence a₁ r
  a 1 + |a 2| + a 3 + |a 4| + a 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2764_276464


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2764_276456

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  c = 5 * a →
  b = 3 * a →
  a = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l2764_276456


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2764_276400

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_calculation (principal interest time : ℚ) 
  (h_principal : principal = 800)
  (h_interest : interest = 160)
  (h_time : time = 4)
  (h_simple_interest : simple_interest principal (5 : ℚ) time = interest) :
  simple_interest principal (5 : ℚ) time = interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2764_276400


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2764_276490

theorem sqrt_meaningful_range (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2764_276490


namespace NUMINAMATH_CALUDE_pizza_cost_is_9_60_l2764_276438

/-- The cost of a single box of pizza -/
def pizza_cost : ℝ := sorry

/-- The cost of a single can of soft drink -/
def soft_drink_cost : ℝ := 2

/-- The cost of a single hamburger -/
def hamburger_cost : ℝ := 3

/-- The number of pizza boxes Robert buys -/
def robert_pizza_boxes : ℕ := 5

/-- The number of soft drink cans Robert buys -/
def robert_soft_drinks : ℕ := 10

/-- The number of hamburgers Teddy buys -/
def teddy_hamburgers : ℕ := 6

/-- The number of soft drink cans Teddy buys -/
def teddy_soft_drinks : ℕ := 10

/-- The total amount spent by Robert and Teddy -/
def total_spent : ℝ := 106

theorem pizza_cost_is_9_60 :
  pizza_cost = 9.60 ∧
  (robert_pizza_boxes : ℝ) * pizza_cost +
  (robert_soft_drinks : ℝ) * soft_drink_cost +
  (teddy_hamburgers : ℝ) * hamburger_cost +
  (teddy_soft_drinks : ℝ) * soft_drink_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_pizza_cost_is_9_60_l2764_276438


namespace NUMINAMATH_CALUDE_robert_ride_time_l2764_276461

/-- The time taken for Robert to ride along a semicircular path on a highway section -/
theorem robert_ride_time :
  let highway_length : ℝ := 1 -- mile
  let highway_width : ℝ := 40 -- feet
  let robert_speed : ℝ := 5 -- miles per hour
  let feet_per_mile : ℝ := 5280
  let path_shape := Semicircle
  let time_taken := 
    (highway_length * feet_per_mile / highway_width) * (π * highway_width / 2) / 
    (robert_speed * feet_per_mile)
  time_taken = π / 10
  := by sorry

end NUMINAMATH_CALUDE_robert_ride_time_l2764_276461


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2764_276495

theorem quadratic_inequality_always_nonnegative :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2764_276495


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2764_276483

/-- The diameter of a circle with area 50.26548245743669 square meters is 8 meters. -/
theorem circle_diameter_from_area :
  let area : Real := 50.26548245743669
  let diameter : Real := 8
  diameter = 2 * Real.sqrt (area / Real.pi) := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2764_276483


namespace NUMINAMATH_CALUDE_reciprocal_of_point_two_l2764_276460

theorem reciprocal_of_point_two (x : ℝ) : x = 0.2 → 1 / x = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_two_l2764_276460


namespace NUMINAMATH_CALUDE_min_value_of_f_l2764_276401

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 1 → f x ≤ f y) ∧
  f x = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2764_276401


namespace NUMINAMATH_CALUDE_randy_baseball_gloves_l2764_276492

theorem randy_baseball_gloves (bats : ℕ) (gloves : ℕ) : 
  bats = 4 → gloves = 7 * bats + 1 → gloves = 29 := by
  sorry

end NUMINAMATH_CALUDE_randy_baseball_gloves_l2764_276492


namespace NUMINAMATH_CALUDE_solution_set_equivalence_a_range_l2764_276489

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x^2 + 1

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ 0}

-- Define the condition that g has two distinct zeros in (1,2)
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₁ ≠ x₂

-- Theorem 1
theorem solution_set_equivalence (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 1} := by sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  has_two_zeros a → -5 < a ∧ a < -2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_a_range_l2764_276489


namespace NUMINAMATH_CALUDE_lance_reading_plan_l2764_276429

/-- Represents the number of pages read on each day -/
structure ReadingPlan where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Checks if a reading plan is valid according to the given conditions -/
def isValidPlan (plan : ReadingPlan) (totalPages : ℕ) : Prop :=
  plan.day2 = plan.day1 - 5 ∧
  plan.day3 = 35 ∧
  plan.day1 + plan.day2 + plan.day3 = totalPages

theorem lance_reading_plan (totalPages : ℕ) (h : totalPages = 100) :
  ∃ (plan : ReadingPlan), isValidPlan plan totalPages ∧ plan.day1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_lance_reading_plan_l2764_276429


namespace NUMINAMATH_CALUDE_xyz_relation_theorem_l2764_276470

/-- A structure representing the relationship between x, y, and z -/
structure XYZRelation where
  x : ℝ
  y : ℝ
  z : ℝ
  c : ℝ
  d : ℝ
  h1 : y^2 = c * z^2  -- y² varies directly with z²
  h2 : y = d / x      -- y varies inversely with x

/-- The theorem statement -/
theorem xyz_relation_theorem (r : XYZRelation) (h3 : r.y = 3) (h4 : r.x = 4) (h5 : r.z = 6) :
  ∃ (r' : XYZRelation), r'.y = 2 ∧ r'.z = 12 ∧ r'.x = 6 ∧ r'.c = r.c ∧ r'.d = r.d :=
sorry


end NUMINAMATH_CALUDE_xyz_relation_theorem_l2764_276470


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2764_276405

theorem fraction_equation_solution :
  ∃ x : ℚ, (5 * x + 3) / (7 * x - 4) = 4128 / 4386 ∧ x = 115 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2764_276405


namespace NUMINAMATH_CALUDE_shoes_cost_calculation_l2764_276425

def budget : ℚ := 200
def shirt_cost : ℚ := 30
def pants_cost : ℚ := 46
def coat_cost : ℚ := 38
def socks_cost : ℚ := 11
def belt_cost : ℚ := 18
def necktie_cost : ℚ := 22
def remaining_money : ℚ := 16

def other_items_cost : ℚ := shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + necktie_cost

theorem shoes_cost_calculation :
  ∃ (shoes_cost : ℚ), 
    shoes_cost = budget - remaining_money - other_items_cost ∧
    shoes_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_shoes_cost_calculation_l2764_276425


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2764_276497

/-- The set of digits to choose from -/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- Predicate to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- Predicate to check if a number is even -/
def is_even (n : Nat) : Bool := n % 2 = 0

/-- The set of four-digit numbers formed from the given digits -/
def valid_numbers : Finset (Fin 10000) :=
  sorry

/-- Theorem stating the number of valid four-digit numbers -/
theorem count_valid_numbers : Finset.card valid_numbers = 180 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2764_276497


namespace NUMINAMATH_CALUDE_rectangle_same_color_l2764_276417

-- Define the color type
def Color := Fin

-- Define a point on the plane
structure Point where
  x : Int
  y : Int

-- Define a coloring function
def coloring (p : Nat) : Point → Color p :=
  sorry

-- The main theorem
theorem rectangle_same_color (p : Nat) :
  ∃ (a b c d : Point), 
    (a.x < b.x ∧ a.y < c.y) ∧ 
    (b.x - a.x = d.x - c.x) ∧ 
    (c.y - a.y = d.y - b.y) ∧
    (coloring p a = coloring p b) ∧
    (coloring p b = coloring p c) ∧
    (coloring p c = coloring p d) :=
  sorry

end NUMINAMATH_CALUDE_rectangle_same_color_l2764_276417


namespace NUMINAMATH_CALUDE_min_students_both_l2764_276496

-- Define the classroom
structure Classroom where
  total : ℕ
  glasses : ℕ
  blue_shirts : ℕ
  both : ℕ

-- Define the conditions
def valid_classroom (c : Classroom) : Prop :=
  c.glasses = (3 * c.total) / 7 ∧
  c.blue_shirts = (4 * c.total) / 9 ∧
  c.both ≤ min c.glasses c.blue_shirts ∧
  c.total ≥ c.glasses + c.blue_shirts - c.both

-- Theorem statement
theorem min_students_both (c : Classroom) (h : valid_classroom c) :
  ∃ (c_min : Classroom), valid_classroom c_min ∧ c_min.both = 8 ∧
  ∀ (c' : Classroom), valid_classroom c' → c'.both ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_students_both_l2764_276496


namespace NUMINAMATH_CALUDE_small_cube_edge_length_small_cube_edge_length_proof_l2764_276431

/-- Given a cube made of 8 smaller cubes with a total volume of 1000 cm³,
    the length of one edge of a smaller cube is 5 cm. -/
theorem small_cube_edge_length : ℝ :=
  let total_volume : ℝ := 1000
  let num_small_cubes : ℕ := 8
  let edge_ratio : ℝ := 2  -- ratio of large cube edge to small cube edge
  
  -- Define the volume of the large cube in terms of the small cube's edge length
  let large_cube_volume (small_edge : ℝ) : ℝ := (edge_ratio * small_edge) ^ 3
  
  -- Define the equation: large cube volume equals total volume
  let volume_equation (small_edge : ℝ) : Prop := large_cube_volume small_edge = total_volume
  
  -- The length of one edge of the smaller cube
  5

/-- Proof of the theorem -/
theorem small_cube_edge_length_proof : small_cube_edge_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_small_cube_edge_length_proof_l2764_276431


namespace NUMINAMATH_CALUDE_ben_age_is_five_l2764_276471

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Four years ago, Chris was twice as old as Amy was then
  ages.chris - 4 = 2 * (ages.amy - 4) ∧
  -- In 5 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 5 = 3 / 4 * (ages.amy + 5)

/-- The theorem to be proved -/
theorem ben_age_is_five :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.ben = 5 := by
  sorry

end NUMINAMATH_CALUDE_ben_age_is_five_l2764_276471


namespace NUMINAMATH_CALUDE_codecracker_combinations_l2764_276422

/-- The number of available colors for the CodeCracker game -/
def num_colors : ℕ := 7

/-- The number of slots in the master code -/
def code_length : ℕ := 5

/-- The number of different master codes that can be formed in the CodeCracker game -/
def num_codes : ℕ := num_colors ^ code_length

theorem codecracker_combinations : num_codes = 16807 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l2764_276422


namespace NUMINAMATH_CALUDE_sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l2764_276449

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem sum_squares_two_odds_not_perfect_square (a b : ℤ) (ha : is_odd a) (hb : is_odd b) :
  ¬∃ n : ℤ, a^2 + b^2 = n^2 := by sorry

theorem sum_squares_three_odds_not_perfect_square (a b c : ℤ) (ha : is_odd a) (hb : is_odd b) (hc : is_odd c) :
  ¬∃ m : ℤ, a^2 + b^2 + c^2 = m^2 := by sorry

end NUMINAMATH_CALUDE_sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l2764_276449


namespace NUMINAMATH_CALUDE_unique_positive_x_exists_l2764_276419

/-- Given a > b > 0, there exists a unique positive x such that 
    f(x) = ((a^(1/3) + b^(1/3)) / 2)^3, where f(x) = (2(a+b)x + 2ab) / (4x + a + b) -/
theorem unique_positive_x_exists (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃! x : ℝ, x > 0 ∧ (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_x_exists_l2764_276419


namespace NUMINAMATH_CALUDE_converse_statement_l2764_276462

theorem converse_statement (x : ℝ) : 
  (∀ x, x ≥ 1 → x^2 + 3*x - 2 ≥ 0) →
  (∀ x, x^2 + 3*x - 2 < 0 → x < 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_statement_l2764_276462


namespace NUMINAMATH_CALUDE_celyna_candy_purchase_l2764_276488

/-- Prove that given the conditions of Celyna's candy purchase, the amount of candy B is 500 grams -/
theorem celyna_candy_purchase (candy_a_weight : ℝ) (candy_a_cost : ℝ) (candy_b_cost : ℝ) (average_price : ℝ) :
  candy_a_weight = 300 →
  candy_a_cost = 5 →
  candy_b_cost = 7 →
  average_price = 1.5 →
  ∃ x : ℝ, x = 500 ∧ 
    (candy_a_cost + candy_b_cost) / ((candy_a_weight + x) / 100) = average_price :=
by sorry

end NUMINAMATH_CALUDE_celyna_candy_purchase_l2764_276488


namespace NUMINAMATH_CALUDE_ages_sum_l2764_276445

theorem ages_sum (diane_future_age diane_current_age : ℕ) 
  (h1 : diane_future_age = 30)
  (h2 : diane_current_age = 16) : ∃ (alex_age allison_age : ℕ), 
  (diane_future_age = alex_age / 2) ∧ 
  (diane_future_age = allison_age * 2) ∧
  (alex_age + allison_age = 47) :=
by sorry

end NUMINAMATH_CALUDE_ages_sum_l2764_276445


namespace NUMINAMATH_CALUDE_system_solution_l2764_276486

-- Define the system of equations
def system (x : Fin 6 → ℚ) : Prop :=
  2 * x 0 + 2 * x 1 - x 2 + x 3 + 4 * x 5 = 0 ∧
  x 0 + 2 * x 1 + 2 * x 2 + 3 * x 4 + x 5 = -2 ∧
  x 0 - 2 * x 1 + x 3 + 2 * x 4 = 0

-- Define the solution
def solution (x : Fin 6 → ℚ) : Prop :=
  x 0 = -1/4 - 5/8 * x 3 - 9/8 * x 4 - 9/8 * x 5 ∧
  x 1 = -1/8 + 3/16 * x 3 - 7/16 * x 4 + 9/16 * x 5 ∧
  x 2 = -3/4 + 1/8 * x 3 - 11/8 * x 4 + 5/8 * x 5

-- Theorem statement
theorem system_solution :
  ∀ x : Fin 6 → ℚ, system x ↔ solution x :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2764_276486


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2764_276478

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 3 + a 5 = 6 →
  a 5 + a 7 + a 9 = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2764_276478


namespace NUMINAMATH_CALUDE_num_planes_is_one_or_three_l2764_276472

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  distinct : point1 ≠ point2

/-- Three pairwise parallel lines -/
structure ThreeParallelLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D
  parallel12 : line1.point2 - line1.point1 = line2.point2 - line2.point1
  parallel23 : line2.point2 - line2.point1 = line3.point2 - line3.point1
  parallel31 : line3.point2 - line3.point1 = line1.point2 - line1.point1

/-- The number of planes determined by three pairwise parallel lines -/
def num_planes_from_parallel_lines (lines : ThreeParallelLines) : Fin 4 :=
  sorry

/-- Theorem: The number of planes determined by three pairwise parallel lines is either 1 or 3 -/
theorem num_planes_is_one_or_three (lines : ThreeParallelLines) :
  (num_planes_from_parallel_lines lines = 1) ∨ (num_planes_from_parallel_lines lines = 3) :=
sorry

end NUMINAMATH_CALUDE_num_planes_is_one_or_three_l2764_276472


namespace NUMINAMATH_CALUDE_triangle_vector_parallel_l2764_276491

theorem triangle_vector_parallel (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a, Real.sqrt 3 * b) = (Real.cos A, Real.sin B) →
  A = π / 3 ∧
  (a = 2 → 2 < b + c ∧ b + c ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_parallel_l2764_276491


namespace NUMINAMATH_CALUDE_max_beads_is_27_l2764_276432

/-- Represents the maximum number of weighings allowed -/
def max_weighings : ℕ := 3

/-- Represents the number of groups in each weighing -/
def groups_per_weighing : ℕ := 3

/-- Calculates the maximum number of beads that can be in the pile -/
def max_beads : ℕ := groups_per_weighing ^ max_weighings

/-- Theorem stating that the maximum number of beads is 27 -/
theorem max_beads_is_27 : max_beads = 27 := by
  sorry

#eval max_beads -- Should output 27

end NUMINAMATH_CALUDE_max_beads_is_27_l2764_276432


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l2764_276499

theorem chord_length_in_circle (r d c : ℝ) (hr : r = 3) (hd : d = 2) :
  r^2 = d^2 + (c/2)^2 → c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l2764_276499


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2764_276430

theorem average_of_three_numbers (A B C : ℝ) 
  (sum_AB : A + B = 147)
  (sum_BC : B + C = 123)
  (sum_AC : A + C = 132) :
  (A + B + C) / 3 = 67 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2764_276430


namespace NUMINAMATH_CALUDE_orange_distribution_l2764_276482

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l2764_276482


namespace NUMINAMATH_CALUDE_weight_of_three_moles_l2764_276447

/-- Given a compound with molecular weight of 882 g/mol, 
    prove that 3 moles of this compound weigh 2646 grams. -/
theorem weight_of_three_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 882 → moles = 3 → moles * molecular_weight = 2646 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_l2764_276447


namespace NUMINAMATH_CALUDE_range_of_f_l2764_276451

def f (x : ℝ) : ℝ := |x| + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2764_276451


namespace NUMINAMATH_CALUDE_point_movement_and_linear_function_l2764_276479

theorem point_movement_and_linear_function (k : ℝ) : 
  let initial_point : ℝ × ℝ := (5, 3)
  let new_point : ℝ × ℝ := (initial_point.1 - 4, initial_point.2 - 1)
  new_point.2 = k * new_point.1 - 2 → k = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_point_movement_and_linear_function_l2764_276479


namespace NUMINAMATH_CALUDE_area_swept_by_small_square_l2764_276450

/-- The area swept by a small square sliding along three sides of a larger square -/
theorem area_swept_by_small_square (large_side small_side : ℝ) :
  large_side > 0 ∧ small_side > 0 ∧ large_side > small_side →
  let swept_area := large_side^2 - (large_side - 2*small_side)^2
  swept_area = 36 ∧ large_side = 10 ∧ small_side = 1 := by
  sorry

#check area_swept_by_small_square

end NUMINAMATH_CALUDE_area_swept_by_small_square_l2764_276450


namespace NUMINAMATH_CALUDE_box_cubes_count_l2764_276441

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height) / cube_volume

/-- Theorem: The minimum number of 3 cubic cm cubes required to build a box
    with dimensions 12 cm × 16 cm × 6 cm is 384. -/
theorem box_cubes_count :
  min_cubes 12 16 6 3 = 384 := by
  sorry

end NUMINAMATH_CALUDE_box_cubes_count_l2764_276441
