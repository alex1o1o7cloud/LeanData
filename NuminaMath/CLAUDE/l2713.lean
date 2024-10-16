import Mathlib

namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l2713_271392

theorem square_of_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 164*x + c = (x + a)^2) → c = 6724 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l2713_271392


namespace NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2713_271337

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem max_sum_arithmetic_sequence :
  let a₁ : ℚ := 5
  let d : ℚ := -5/7
  let S : ℕ → ℚ := λ n => sum_arithmetic_sequence a₁ d n
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧
    (∀ m : ℕ, S m ≤ S n) ∧
    S n = 1075/14 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_arithmetic_sequence_l2713_271337


namespace NUMINAMATH_CALUDE_tan_period_l2713_271322

/-- The smallest positive period of tan((a + b)x/2) given conditions -/
theorem tan_period (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = 1) : 
  let f := fun x => Real.tan ((a + b) * x / 2)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l2713_271322


namespace NUMINAMATH_CALUDE_area_square_on_hypotenuse_for_24cm_l2713_271366

/-- An isosceles right triangle with an inscribed square -/
structure TriangleWithSquare where
  /-- Side length of the inscribed square touching the right angle -/
  s : ℝ
  /-- The square touches the right angle vertex -/
  touches_right_angle : s > 0
  /-- The opposite side of the square is parallel to the hypotenuse -/
  parallel_to_hypotenuse : True

/-- The area of a square inscribed along the hypotenuse of the triangle -/
def area_square_on_hypotenuse (t : TriangleWithSquare) : ℝ :=
  t.s ^ 2

theorem area_square_on_hypotenuse_for_24cm (t : TriangleWithSquare) 
  (h : t.s = 24) : area_square_on_hypotenuse t = 576 := by
  sorry

end NUMINAMATH_CALUDE_area_square_on_hypotenuse_for_24cm_l2713_271366


namespace NUMINAMATH_CALUDE_square_of_one_forty_four_l2713_271383

/-- Represents a number in a given base -/
def BaseRepresentation (n : ℕ) (b : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ n = d₁ * b^2 + d₂ * b + d₃

/-- The number 144 in base b -/
def OneFortyFour (b : ℕ) : ℕ := b^2 + 4*b + 4

theorem square_of_one_forty_four (b : ℕ) :
  b > 4 →
  BaseRepresentation (OneFortyFour b) b →
  ∃ k : ℕ, OneFortyFour b = k^2 :=
sorry

end NUMINAMATH_CALUDE_square_of_one_forty_four_l2713_271383


namespace NUMINAMATH_CALUDE_total_watching_time_l2713_271394

def first_show_length : ℕ := 30
def second_show_multiplier : ℕ := 4

theorem total_watching_time :
  first_show_length + first_show_length * second_show_multiplier = 150 :=
by sorry

end NUMINAMATH_CALUDE_total_watching_time_l2713_271394


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2713_271373

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2713_271373


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2713_271381

theorem algebraic_expression_value : 
  let x : ℝ := Real.sqrt 3 + 2
  (x^2 - 4*x + 3) = 2 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2713_271381


namespace NUMINAMATH_CALUDE_football_progress_l2713_271380

/-- Calculates the net progress of a football team given a loss and a gain in yards. -/
def net_progress (loss : ℤ) (gain : ℤ) : ℤ := -loss + gain

/-- Theorem stating that a loss of 5 yards followed by a gain of 8 yards results in a net progress of 3 yards. -/
theorem football_progress : net_progress 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l2713_271380


namespace NUMINAMATH_CALUDE_test_to_quiz_ratio_l2713_271378

def total_points : ℕ := 265
def homework_points : ℕ := 40

def quiz_points : ℕ := homework_points + 5

def test_points : ℕ := total_points - quiz_points - homework_points

theorem test_to_quiz_ratio :
  (test_points : ℚ) / quiz_points = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_test_to_quiz_ratio_l2713_271378


namespace NUMINAMATH_CALUDE_trioball_playing_time_l2713_271335

theorem trioball_playing_time (num_children : ℕ) (game_duration : ℕ) (players_per_game : ℕ) :
  num_children = 3 →
  game_duration = 120 →
  players_per_game = 2 →
  ∃ (individual_time : ℕ),
    individual_time * num_children = players_per_game * game_duration ∧
    individual_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_trioball_playing_time_l2713_271335


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2713_271370

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 1 = 0 ∧ x₂^2 - 1 = 0 ∧ x₁ = 1 ∧ x₂ = -1) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 ∧
    x₁ = (3 + Real.sqrt 5) / 2 ∧ x₂ = (3 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2713_271370


namespace NUMINAMATH_CALUDE_probability_sum_eight_l2713_271341

/-- A fair die with 6 faces -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing two dice -/
structure TwoDiceThrow :=
  (die1 : Die)
  (die2 : Die)

/-- The sum of the numbers on two dice -/
def diceSum (throw : TwoDiceThrow) : Nat :=
  throw.die1.faces.val + 1 + throw.die2.faces.val + 1

/-- The set of all possible throws of two dice -/
def allThrows : Finset TwoDiceThrow :=
  sorry

/-- The set of throws where the sum is 8 -/
def sumEightThrows : Finset TwoDiceThrow :=
  sorry

/-- The probability of an event occurring when throwing two fair dice -/
def probability (event : Finset TwoDiceThrow) : Rat :=
  (event.card : Rat) / (allThrows.card : Rat)

theorem probability_sum_eight :
  probability sumEightThrows = 5 / 36 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_eight_l2713_271341


namespace NUMINAMATH_CALUDE_problem_solution_l2713_271353

theorem problem_solution : 
  ((-1)^2023 - Real.sqrt (2 + 1/4) + ((-1 : ℝ)^(1/3 : ℝ)) + 1/2 = -3) ∧ 
  (2 * Real.sqrt 3 + |1 - Real.sqrt 3| - (-1)^2022 + 2 = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2713_271353


namespace NUMINAMATH_CALUDE_carlsons_land_cost_l2713_271377

def land_problem (initial_land : ℝ) (final_land : ℝ) (price_per_sqm : ℝ) (additional_cost : ℝ) : Prop :=
  let additional_land := final_land - initial_land
  let additional_land_cost := additional_land * price_per_sqm
  let first_land_cost := additional_land_cost - additional_cost
  first_land_cost = 8000

theorem carlsons_land_cost :
  land_problem 300 900 20 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_carlsons_land_cost_l2713_271377


namespace NUMINAMATH_CALUDE_no_solution_equations_l2713_271338

theorem no_solution_equations :
  (∀ x : ℝ, |4*x| + 7 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (-3*x) + 1 ≠ 0) ∧
  (∃ x : ℝ, (x - 3)^2 = 0) ∧
  (∃ x : ℝ, Real.sqrt (2*x) - 5 = 0) ∧
  (∃ x : ℝ, |2*x| - 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equations_l2713_271338


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2713_271344

/-- The polynomial z^5 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^5 - z^3 + 1

/-- n-th roots of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

theorem smallest_n_for_roots_of_unity :
  (∀ n : ℕ, n > 0 → all_roots_are_nth_roots_of_unity n → n ≥ 15) ∧
  all_roots_are_nth_roots_of_unity 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l2713_271344


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2713_271393

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and
    common difference d = 1, for the first 2k terms. -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := 2 * k
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + k := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2713_271393


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2713_271303

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2713_271303


namespace NUMINAMATH_CALUDE_baron_munchausen_crowd_size_l2713_271304

theorem baron_munchausen_crowd_size :
  ∃ (n : ℕ), n > 0 ∧
  (n / 2 + n / 3 + n / 5 ≤ n + 1) ∧
  (∀ m : ℕ, m > n → m / 2 + m / 3 + m / 5 > m + 1) ∧
  n = 37 := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_crowd_size_l2713_271304


namespace NUMINAMATH_CALUDE_inverse_71_mod_83_l2713_271340

theorem inverse_71_mod_83 (h : (17⁻¹ : ZMod 83) = 53) : (71⁻¹ : ZMod 83) = 53 := by
  sorry

end NUMINAMATH_CALUDE_inverse_71_mod_83_l2713_271340


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l2713_271371

theorem cricket_innings_problem (initial_average : ℝ) (next_innings_runs : ℝ) (average_increase : ℝ) :
  initial_average = 32 →
  next_innings_runs = 137 →
  average_increase = 5 →
  ∃ n : ℕ, (n : ℝ) * initial_average + next_innings_runs = (n + 1 : ℝ) * (initial_average + average_increase) ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l2713_271371


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2713_271333

theorem complex_equation_solution (x y : ℝ) :
  (Complex.I * (x + Complex.I) + y = 1 + 2 * Complex.I) →
  x - y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2713_271333


namespace NUMINAMATH_CALUDE_shooting_training_equivalence_l2713_271387

-- Define the propositions
variable (p q : Prop)

-- Define "both shots hit the target"
def both_hit (p q : Prop) : Prop := p ∧ q

-- Define "exactly one shot hits the target"
def exactly_one_hit (p q : Prop) : Prop := (p ∧ ¬q) ∨ (¬p ∧ q)

-- Theorem stating the equivalence
theorem shooting_training_equivalence :
  (both_hit p q ↔ p ∧ q) ∧
  (exactly_one_hit p q ↔ (p ∧ ¬q) ∨ (¬p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_shooting_training_equivalence_l2713_271387


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2713_271398

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → (x + y = r + 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      ((x'^2 + y'^2 - r^2) * ((x' + y') - (r + 1)) ≥ 0))) →
  r = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2713_271398


namespace NUMINAMATH_CALUDE_last_box_contents_l2713_271368

-- Define the total number of bars for each type of chocolate
def total_A : ℕ := 853845
def total_B : ℕ := 537896
def total_C : ℕ := 729763

-- Define the box capacity for each type of chocolate
def capacity_A : ℕ := 9
def capacity_B : ℕ := 11
def capacity_C : ℕ := 15

-- Theorem to prove the number of bars in the last partially filled box for each type
theorem last_box_contents :
  (total_A % capacity_A = 4) ∧
  (total_B % capacity_B = 3) ∧
  (total_C % capacity_C = 8) := by
  sorry

end NUMINAMATH_CALUDE_last_box_contents_l2713_271368


namespace NUMINAMATH_CALUDE_xu_jun_current_age_l2713_271346

-- Define Xu Jun's current age
def xu_jun_age : ℕ := sorry

-- Define the teacher's current age
def teacher_age : ℕ := sorry

-- Condition 1: Two years ago, the teacher's age was 3 times Xu Jun's age
axiom condition1 : teacher_age - 2 = 3 * (xu_jun_age - 2)

-- Condition 2: In 8 years, the teacher's age will be twice Xu Jun's age
axiom condition2 : teacher_age + 8 = 2 * (xu_jun_age + 8)

-- Theorem to prove
theorem xu_jun_current_age : xu_jun_age = 12 := by sorry

end NUMINAMATH_CALUDE_xu_jun_current_age_l2713_271346


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l2713_271349

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  ∃ (meal_cost : ℚ), meal_cost * (adults + children) = total_bill ∧ meal_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l2713_271349


namespace NUMINAMATH_CALUDE_cartons_in_load_l2713_271342

/-- Represents the weight of vegetables in a store's delivery truck. -/
structure VegetableLoad where
  crate_weight : ℕ
  carton_weight : ℕ
  num_crates : ℕ
  total_weight : ℕ

/-- Calculates the number of cartons in a vegetable load. -/
def num_cartons (load : VegetableLoad) : ℕ :=
  (load.total_weight - load.crate_weight * load.num_crates) / load.carton_weight

/-- Theorem stating that the number of cartons in the specific load is 16. -/
theorem cartons_in_load : 
  let load : VegetableLoad := {
    crate_weight := 4,
    carton_weight := 3,
    num_crates := 12,
    total_weight := 96
  }
  num_cartons load = 16 := by
  sorry


end NUMINAMATH_CALUDE_cartons_in_load_l2713_271342


namespace NUMINAMATH_CALUDE_primitive_root_mod_p_squared_l2713_271313

theorem primitive_root_mod_p_squared (p : Nat) (x : Nat) 
  (h_p : Nat.Prime p) 
  (h_p_odd : Odd p) 
  (h_x_prim_root : IsPrimitiveRoot x p) : 
  IsPrimitiveRoot x (p^2) ∨ IsPrimitiveRoot (x + p) (p^2) := by
  sorry

end NUMINAMATH_CALUDE_primitive_root_mod_p_squared_l2713_271313


namespace NUMINAMATH_CALUDE_pears_left_l2713_271315

theorem pears_left (keith_picked mike_picked keith_gave_away : ℕ) 
  (h1 : keith_picked = 47)
  (h2 : mike_picked = 12)
  (h3 : keith_gave_away = 46) :
  keith_picked - keith_gave_away + mike_picked = 13 := by
sorry

end NUMINAMATH_CALUDE_pears_left_l2713_271315


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2713_271328

theorem exponent_multiplication (x : ℝ) (a b : ℕ) :
  x^a * x^b = x^(a + b) := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2713_271328


namespace NUMINAMATH_CALUDE_inequality_proof_l2713_271302

theorem inequality_proof (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2713_271302


namespace NUMINAMATH_CALUDE_box_fits_cubes_l2713_271355

/-- A rectangular box with given dimensions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A cube with a given volume -/
structure Cube where
  volume : ℝ

/-- The number of cubes that can fit in the box -/
def cubes_fit (b : Box) (c : Cube) : ℕ := 24

theorem box_fits_cubes (b : Box) (c : Cube) :
  b.length = 9 ∧ b.width = 8 ∧ b.height = 12 ∧ c.volume = 27 →
  cubes_fit b c = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_fits_cubes_l2713_271355


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2713_271389

theorem quadratic_equation_condition (m : ℝ) : 
  (m ^ 2 - 7 = 2 ∧ m - 3 ≠ 0) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2713_271389


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l2713_271399

theorem smallest_integer_solution (m : ℚ) : 
  (∃ x : ℤ, (3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1) ∧ 
    (∀ y : ℤ, 3 * (y + 1) - 2 ≤ 4 * (y - 3) + 1 → x ≤ y) ∧
    ((1 : ℚ) / 2 * x - m = 5)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l2713_271399


namespace NUMINAMATH_CALUDE_fifth_invoice_number_l2713_271350

/-- Represents the systematic sampling process for invoices -/
def systematicSampling (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * interval

/-- Theorem stating that the fifth sampled invoice number is 215 -/
theorem fifth_invoice_number :
  systematicSampling 15 50 5 = 215 := by
  sorry

end NUMINAMATH_CALUDE_fifth_invoice_number_l2713_271350


namespace NUMINAMATH_CALUDE_prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l2713_271330

/-- The probability of rolling a sum of 32 with four fair eight-sided dice -/
theorem prob_sum_32_four_eight_sided_dice : ℝ :=
  let num_faces : ℕ := 8
  let num_dice : ℕ := 4
  let target_sum : ℕ := 32
  let prob_max_face : ℝ := 1 / num_faces
  (prob_max_face ^ num_dice : ℝ)

#check prob_sum_32_four_eight_sided_dice

theorem prob_sum_32_four_eight_sided_dice_eq_frac :
  prob_sum_32_four_eight_sided_dice = 1 / 4096 := by sorry

end NUMINAMATH_CALUDE_prob_sum_32_four_eight_sided_dice_prob_sum_32_four_eight_sided_dice_eq_frac_l2713_271330


namespace NUMINAMATH_CALUDE_ball_count_proof_l2713_271314

theorem ball_count_proof (a : ℕ) (h1 : a > 0) (h2 : 3 ≤ a) : 
  (3 : ℚ) / a = 1 / 4 → a = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_count_proof_l2713_271314


namespace NUMINAMATH_CALUDE_sum_seven_more_likely_than_eight_l2713_271345

def dice_sum_probability (sum : Nat) : Rat :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range 6) (Finset.range 6))).card / 36

theorem sum_seven_more_likely_than_eight :
  dice_sum_probability 7 > dice_sum_probability 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_more_likely_than_eight_l2713_271345


namespace NUMINAMATH_CALUDE_triangle_area_l2713_271356

/-- The area of a triangle with sides 9, 40, and 41 is 180 -/
theorem triangle_area : ℝ → Prop := fun area =>
  let a := 9
  let b := 40
  let c := 41
  (a * a + b * b = c * c) → (area = (1 / 2) * a * b)

#check triangle_area 180

end NUMINAMATH_CALUDE_triangle_area_l2713_271356


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2713_271362

/-- Definition of a double subtraction point -/
def is_double_subtraction_point (k b x y : ℝ) : Prop :=
  k ≠ 0 ∧ y = k * x ∧ y = b

/-- The main theorem -/
theorem inequality_system_solution 
  (k : ℝ) 
  (h_k : k ≠ 0)
  (a : ℝ)
  (h_double_sub : is_double_subtraction_point k (a - 2) 3 (3 * k)) :
  {y : ℝ | 2 * (y + 1) < 5 * y - 7 ∧ (y + a) / 2 < 5} = {y : ℝ | 3 < y ∧ y < 8} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2713_271362


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2713_271375

theorem no_integer_solutions :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2713_271375


namespace NUMINAMATH_CALUDE_awards_distribution_l2713_271336

/-- The number of ways to distribute n distinct awards to k students, where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 3 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l2713_271336


namespace NUMINAMATH_CALUDE_morgan_pens_count_l2713_271347

/-- The number of pens Morgan has -/
def total_pens (red blue black green purple : ℕ) : ℕ :=
  red + blue + black + green + purple

/-- Theorem: Morgan has 231 pens in total -/
theorem morgan_pens_count : total_pens 65 45 58 36 27 = 231 := by
  sorry

end NUMINAMATH_CALUDE_morgan_pens_count_l2713_271347


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2713_271301

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2713_271301


namespace NUMINAMATH_CALUDE_min_value_problem_l2713_271357

theorem min_value_problem (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2*y = 1) :
  ∃ m : ℝ, m = 8/9 ∧ ∀ x' y' : ℝ, x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2713_271357


namespace NUMINAMATH_CALUDE_cat_stickers_count_l2713_271324

theorem cat_stickers_count (space_stickers : Nat) (friends : Nat) (leftover : Nat) (cat_stickers : Nat) : 
  space_stickers = 100 →
  friends = 3 →
  leftover = 3 →
  (space_stickers + cat_stickers - leftover) % friends = 0 →
  cat_stickers = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_stickers_count_l2713_271324


namespace NUMINAMATH_CALUDE_ellipse_equation_l2713_271384

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) ∧ 
  (2*a*b = 4) ∧
  (∃ (c : ℝ), a^2 - b^2 = c^2 ∧ c = Real.sqrt 3) →
  (∃ (x y : ℝ), x^2/4 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2713_271384


namespace NUMINAMATH_CALUDE_inequality_proof_l2713_271361

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : 0 < b) (hb1 : b < 1) :
  ab^2 > ab ∧ ab > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2713_271361


namespace NUMINAMATH_CALUDE_probability_two_not_selected_l2713_271396

theorem probability_two_not_selected (S : Finset Nat) (a b : Nat) 
  (h1 : S.card = 4) (h2 : a ∈ S) (h3 : b ∈ S) (h4 : a ≠ b) :
  (Finset.filter (λ T : Finset Nat => T.card = 2 ∧ a ∉ T ∧ b ∉ T) (S.powerset)).card / (Finset.filter (λ T : Finset Nat => T.card = 2) (S.powerset)).card = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_not_selected_l2713_271396


namespace NUMINAMATH_CALUDE_eighteen_bottles_needed_l2713_271360

/-- Calculates the minimum number of small bottles needed to fill a large bottle and a vase -/
def minimum_bottles (small_capacity : ℕ) (large_capacity : ℕ) (vase_capacity : ℕ) : ℕ :=
  let large_bottles := large_capacity / small_capacity
  let remaining_for_vase := vase_capacity
  let vase_bottles := (remaining_for_vase + small_capacity - 1) / small_capacity
  large_bottles + vase_bottles

/-- Theorem stating that 18 small bottles are needed to fill the large bottle and vase -/
theorem eighteen_bottles_needed :
  minimum_bottles 45 675 95 = 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_bottles_needed_l2713_271360


namespace NUMINAMATH_CALUDE_lcm_6_15_l2713_271358

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l2713_271358


namespace NUMINAMATH_CALUDE_cake_measuring_l2713_271308

theorem cake_measuring (flour_needed : ℚ) (milk_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 10/3 ∧ milk_needed = 3/2 ∧ cup_capacity = 1/3 → 
  Int.ceil (flour_needed / cup_capacity) + Int.ceil (milk_needed / cup_capacity) = 15 := by
sorry

end NUMINAMATH_CALUDE_cake_measuring_l2713_271308


namespace NUMINAMATH_CALUDE_sum_of_three_digit_permutations_not_2018_l2713_271331

theorem sum_of_three_digit_permutations_not_2018 (a b c : ℕ) : 
  (0 < a ∧ a ≤ 9) → (0 < b ∧ b ≤ 9) → (0 < c ∧ c ≤ 9) → 
  a ≠ b → b ≠ c → a ≠ c →
  (100*a + 10*b + c) + (100*a + 10*c + b) + (100*b + 10*a + c) + 
  (100*b + 10*c + a) + (100*c + 10*a + b) + (100*c + 10*b + a) ≠ 2018 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_digit_permutations_not_2018_l2713_271331


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l2713_271306

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l2713_271306


namespace NUMINAMATH_CALUDE_chessboard_parallelogram_l2713_271374

/-- Represents a chess piece placement on an n×n board -/
structure Placement (n : ℕ) where
  pieces : Finset (Fin n × Fin n)

/-- Checks if four pieces form a parallelogram -/
def is_parallelogram (n : ℕ) (p1 p2 p3 p4 : Fin n × Fin n) : Prop :=
  (p1.1 + p3.1 = p2.1 + p4.1) ∧ (p1.2 + p3.2 = p2.2 + p4.2)

/-- The main theorem about chess piece placements -/
theorem chessboard_parallelogram (n : ℕ) (h : n > 1) :
  (∀ (p : Placement n), p.pieces.card = 2 * n →
    ∃ (p1 p2 p3 p4 : Fin n × Fin n),
      p1 ∈ p.pieces ∧ p2 ∈ p.pieces ∧ p3 ∈ p.pieces ∧ p4 ∈ p.pieces ∧
      is_parallelogram n p1 p2 p3 p4) ∧
  (∃ (p : Placement n), p.pieces.card = 2 * n - 1 ∧
    ∀ (p1 p2 p3 p4 : Fin n × Fin n),
      p1 ∈ p.pieces → p2 ∈ p.pieces → p3 ∈ p.pieces → p4 ∈ p.pieces →
      ¬is_parallelogram n p1 p2 p3 p4) :=
sorry

end NUMINAMATH_CALUDE_chessboard_parallelogram_l2713_271374


namespace NUMINAMATH_CALUDE_tournament_theorem_l2713_271382

/-- A tournament is a complete directed graph -/
structure Tournament (n : ℕ) where
  edges : Fin n → Fin n → Bool
  complete : ∀ i j, i ≠ j → edges i j ≠ edges j i
  no_self_edges : ∀ i, edges i i = false

/-- A set of edges in a tournament -/
def EdgeSet (n : ℕ) := Fin n → Fin n → Bool

/-- Reverse the orientation of edges in the given set -/
def reverseEdges (T : Tournament n) (S : EdgeSet n) : Tournament n where
  edges i j := if S i j then ¬(T.edges i j) else T.edges i j
  complete := sorry
  no_self_edges := sorry

/-- A graph contains a cycle -/
def hasCycle (T : Tournament n) : Prop := sorry

/-- A graph is acyclic -/
def isAcyclic (T : Tournament n) : Prop := ¬(hasCycle T)

/-- The number of edges in an edge set -/
def edgeCount (S : EdgeSet n) : ℕ := sorry

theorem tournament_theorem (n : ℕ) (h : n = 8) :
  (∃ T : Tournament n, ∀ S : EdgeSet n, edgeCount S ≤ 7 → hasCycle (reverseEdges T S)) ∧
  (∀ T : Tournament n, ∃ S : EdgeSet n, edgeCount S ≤ 8 ∧ isAcyclic (reverseEdges T S)) :=
sorry

end NUMINAMATH_CALUDE_tournament_theorem_l2713_271382


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2713_271327

theorem max_sum_of_squares (a b c : ℝ) 
  (h1 : a + b = c - 1) 
  (h2 : a * b = c^2 - 7*c + 14) : 
  ∃ (m : ℝ), (∀ (x y z : ℝ), x + y = z - 1 → x * y = z^2 - 7*z + 14 → x^2 + y^2 ≤ m) ∧ a^2 + b^2 = m :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2713_271327


namespace NUMINAMATH_CALUDE_trout_percentage_is_sixty_percent_l2713_271359

def total_fish : ℕ := 5
def trout_price : ℕ := 5
def bluegill_price : ℕ := 4
def sunday_earnings : ℕ := 23

theorem trout_percentage_is_sixty_percent :
  ∃ (trout blue_gill : ℕ),
    trout + blue_gill = total_fish ∧
    trout * trout_price + blue_gill * bluegill_price = sunday_earnings ∧
    (trout : ℚ) / (total_fish : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trout_percentage_is_sixty_percent_l2713_271359


namespace NUMINAMATH_CALUDE_quadratic_decreasing_threshold_l2713_271332

/-- Represents a quadratic function of the form ax^2 - 2ax + 1 -/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

/-- Proves that for a quadratic function f(x) = ax^2 - 2ax + 1 where a < 0,
    the minimum value of m for which f(x) is decreasing for all x > m is 1 -/
theorem quadratic_decreasing_threshold (a : ℝ) (h : a < 0) :
  ∃ m : ℝ, m = 1 ∧ ∀ x > m, ∀ y > x,
    QuadraticFunction a y < QuadraticFunction a x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_threshold_l2713_271332


namespace NUMINAMATH_CALUDE_angle_problem_l2713_271363

theorem angle_problem (angle1 angle2 angle3 angle4 angle5 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle5 →
  angle3 + angle4 = 180 →
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l2713_271363


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2713_271311

/-- The number of diagonals from one vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l2713_271311


namespace NUMINAMATH_CALUDE_cos_pi_minus_theta_l2713_271310

theorem cos_pi_minus_theta (θ : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.cos (π - θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_theta_l2713_271310


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2713_271388

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2713_271388


namespace NUMINAMATH_CALUDE_original_number_l2713_271334

theorem original_number : ∃ x : ℤ, 63 - 2 * x = 51 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2713_271334


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2713_271319

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (B < 5) ∧                        -- B is a digit in base 5
  (c > 7) ∧                        -- c is a base greater than 7
  (31 * B = 4 * c + 4) ∧           -- BBB_5 = 44_c
  (∀ (B' c' : ℕ),                  -- For all other valid B' and c'
    (B' < 5) →
    (c' > 7) →
    (31 * B' = 4 * c' + 4) →
    (B + c ≤ B' + c')) ∧
  (B + c = 25)                     -- The smallest sum is 25
  := by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2713_271319


namespace NUMINAMATH_CALUDE_smallest_side_difference_l2713_271379

theorem smallest_side_difference (P Q R : ℕ) (h_perimeter : P + Q + R = 3010)
  (h_order : P < Q ∧ Q ≤ R) : ∃ (P' Q' R' : ℕ), 
  P' + Q' + R' = 3010 ∧ P' < Q' ∧ Q' ≤ R' ∧ Q' - P' = 1 ∧ 
  ∀ (X Y Z : ℕ), X + Y + Z = 3010 → X < Y → Y ≤ Z → Y - X ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_smallest_side_difference_l2713_271379


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2713_271367

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.abs ((1 + 3 * i) / (1 - i)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2713_271367


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2713_271390

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → ∃ k : ℕ, 2 ≤ k ∧ k ≤ 12 ∧ m % k ≠ k - 1) ∧
  n = 27719 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2713_271390


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2713_271323

theorem simplify_trig_expression (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = Real.cos (α - β) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2713_271323


namespace NUMINAMATH_CALUDE_fraction_and_percentage_l2713_271372

theorem fraction_and_percentage (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_and_percentage_l2713_271372


namespace NUMINAMATH_CALUDE_toms_age_l2713_271351

/-- Tom's age problem -/
theorem toms_age (s t : ℕ) : 
  t = 2 * s - 1 →  -- Tom's age is 1 year less than twice his sister's age
  t + s = 14 →     -- The sum of their ages is 14 years
  t = 9            -- Tom's age is 9 years
:= by sorry

end NUMINAMATH_CALUDE_toms_age_l2713_271351


namespace NUMINAMATH_CALUDE_soccer_team_games_l2713_271339

/-- Calculates the number of games played by a soccer team based on pizza slices and goals scored. -/
theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (goals_per_game : ℕ) 
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : goals_per_game = 9)
  (h4 : pizzas * slices_per_pizza = goals_per_game * (pizzas * slices_per_pizza / goals_per_game)) :
  pizzas * slices_per_pizza / goals_per_game = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l2713_271339


namespace NUMINAMATH_CALUDE_krystianas_apartment_occupancy_l2713_271343

/-- Represents the apartment building owned by Krystiana -/
structure ApartmentBuilding where
  firstFloorCost : ℕ
  secondFloorCost : ℕ
  thirdFloorCost : ℕ
  roomsPerFloor : ℕ
  monthlyEarnings : ℕ

/-- Calculates the number of occupied rooms on the third floor -/
def occupiedThirdFloorRooms (building : ApartmentBuilding) : ℕ :=
  building.roomsPerFloor - (building.roomsPerFloor * building.thirdFloorCost + 
    building.roomsPerFloor * building.secondFloorCost + 
    building.roomsPerFloor * building.firstFloorCost - 
    building.monthlyEarnings) / building.thirdFloorCost

/-- Theorem stating that the number of occupied rooms on the third floor is 2 -/
theorem krystianas_apartment_occupancy 
  (building : ApartmentBuilding) 
  (h1 : building.firstFloorCost = 15)
  (h2 : building.secondFloorCost = 20)
  (h3 : building.thirdFloorCost = 2 * building.firstFloorCost)
  (h4 : building.roomsPerFloor = 3)
  (h5 : building.monthlyEarnings = 165) :
  occupiedThirdFloorRooms building = 2 := by
  sorry

#eval occupiedThirdFloorRooms {
  firstFloorCost := 15,
  secondFloorCost := 20,
  thirdFloorCost := 30,
  roomsPerFloor := 3,
  monthlyEarnings := 165
}

end NUMINAMATH_CALUDE_krystianas_apartment_occupancy_l2713_271343


namespace NUMINAMATH_CALUDE_valid_arrangements_l2713_271397

/-- The number of ways to arrange 7 distinct digits with 1 to the left of 2 and 3 -/
def arrange_digits : ℕ :=
  (Nat.choose 7 3) * (Nat.factorial 4)

/-- Theorem stating that there are 840 valid arrangements -/
theorem valid_arrangements : arrange_digits = 840 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2713_271397


namespace NUMINAMATH_CALUDE_chad_sandwiches_l2713_271312

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights 5 boxes of crackers last Chad -/
def num_nights : ℕ := 56

/-- The number of sandwiches Chad has each night -/
def sandwiches_per_night : ℕ := 5

theorem chad_sandwiches :
  sandwiches_per_night * crackers_per_sandwich * num_nights =
  num_boxes * sleeves_per_box * crackers_per_sleeve :=
sorry

end NUMINAMATH_CALUDE_chad_sandwiches_l2713_271312


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l2713_271320

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := ((x/2)^α) / (x-1)

noncomputable def g (α : ℝ) : ℝ := (deriv^[4] (f α)) 2

theorem leading_coefficient_of_g (α : ℝ) : 
  ∃ (p : Polynomial ℝ), (∀ x, g x = p.eval x) ∧ p.leadingCoeff = 1/16 :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l2713_271320


namespace NUMINAMATH_CALUDE_min_value_expression_l2713_271300

theorem min_value_expression (x y : ℝ) 
  (h1 : |x| < 1) 
  (h2 : |y| < 2) 
  (h3 : x * y = 1) : 
  (1 / (1 - x^2)) + (4 / (4 - y^2)) ≥ 4 ∧ 
  ∃ (x₀ y₀ : ℝ), |x₀| < 1 ∧ |y₀| < 2 ∧ x₀ * y₀ = 1 ∧ 
    (1 / (1 - x₀^2)) + (4 / (4 - y₀^2)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2713_271300


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2713_271395

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → n ≥ 1200 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l2713_271395


namespace NUMINAMATH_CALUDE_f_properties_l2713_271318

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 - 6*x^2 - 9*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-4) 2

-- Theorem statement
theorem f_properties :
  -- 1. f is strictly decreasing on (-∞, -3) and (-1, +∞)
  (∀ x y, x < y → x < -3 → f y < f x) ∧
  (∀ x y, x < y → -1 < x → f y < f x) ∧
  -- 2. The minimum value of f on [-4, 2] is -47
  (∀ x ∈ interval, f x ≥ -47) ∧
  (∃ x ∈ interval, f x = -47) ∧
  -- 3. The maximum value of f on [-4, 2] is 7
  (∀ x ∈ interval, f x ≤ 7) ∧
  (∃ x ∈ interval, f x = 7) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2713_271318


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2713_271354

/-- A rectangle with perimeter 60 and area 221 has a shorter side of length 13 -/
theorem rectangle_shorter_side : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧ 
  2 * x + 2 * y = 60 ∧ 
  x * y = 221 ∧ 
  min x y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2713_271354


namespace NUMINAMATH_CALUDE_share_calculation_l2713_271369

theorem share_calculation (total_amount : ℕ) (ratio_parts : List ℕ) : 
  total_amount = 4800 → 
  ratio_parts = [2, 4, 6] → 
  (total_amount / (ratio_parts.sum)) * (ratio_parts.head!) = 800 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l2713_271369


namespace NUMINAMATH_CALUDE_double_price_increase_l2713_271386

theorem double_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  let first_increase := original_price * (1 + increase_percentage / 100)
  let second_increase := first_increase * (1 + increase_percentage / 100)
  increase_percentage = 15 →
  second_increase = original_price * (1 + 32.25 / 100) :=
by sorry

end NUMINAMATH_CALUDE_double_price_increase_l2713_271386


namespace NUMINAMATH_CALUDE_chord_length_l2713_271352

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 2 * y - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 2}
  let chord := line ∩ circle
  ∃ (a b : ℝ), (a, b) ∈ chord ∧ 
    ∃ (c d : ℝ), (c, d) ∈ chord ∧ 
      (a - c)^2 + (b - d)^2 = (4 * Real.sqrt 3 / 3)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2713_271352


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l2713_271385

theorem largest_n_for_sin_cos_inequality : 
  ∃ n : ℕ+, (∀ m : ℕ+, m > n → ∃ x : ℝ, (Real.sin x + Real.cos x)^(m : ℝ) < 2 / (m : ℝ)) ∧
             (∀ x : ℝ, (Real.sin x + Real.cos x)^(n : ℝ) ≥ 2 / (n : ℝ)) ∧
             n = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l2713_271385


namespace NUMINAMATH_CALUDE_distance_D_to_ABC_plane_l2713_271376

/-- The distance from a point to a plane in 3D space --/
def distancePointToPlane (p : ℝ × ℝ × ℝ) (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance from point D to plane ABC is 11 --/
theorem distance_D_to_ABC_plane : 
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, 1, -2)
  let C : ℝ × ℝ × ℝ := (6, 3, 7)
  let D : ℝ × ℝ × ℝ := (-5, -4, 8)
  distancePointToPlane D A B C = 11 := by sorry

end NUMINAMATH_CALUDE_distance_D_to_ABC_plane_l2713_271376


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2713_271364

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ + a₈ = 6,
    prove that 3a₂ + a₁₆ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 6) : 
  3 * a 2 + a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2713_271364


namespace NUMINAMATH_CALUDE_odd_function_property_l2713_271305

-- Define an odd function f from ℝ to ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+2) = -1/f(x)
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -1 / f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_prop : has_property f) : 
  f 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l2713_271305


namespace NUMINAMATH_CALUDE_total_apples_proof_l2713_271391

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73
def benny_apples : ℕ := 48
def lucy_sales : ℕ := 15

theorem total_apples_proof :
  pinky_apples + danny_apples + benny_apples = 157 :=
by sorry

end NUMINAMATH_CALUDE_total_apples_proof_l2713_271391


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2713_271348

def M : Set Int := {1, 2, 3, 4}
def N : Set Int := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2713_271348


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2713_271309

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = 
  (Real.rpow 9 (1/3)) / (3 * (Real.rpow 9 (1/3) + 1)) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2713_271309


namespace NUMINAMATH_CALUDE_rational_function_value_l2713_271316

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x * (a * x + b)
def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

-- State the theorem
theorem rational_function_value
  (a b : ℝ)
  (h1 : p a b 1 / q 1 = -1)
  (h2 : a + b = 1/4) :
  p a b (-1) / q (-1) = (a - b) / 4 := by
sorry

end NUMINAMATH_CALUDE_rational_function_value_l2713_271316


namespace NUMINAMATH_CALUDE_power_nine_mod_seven_l2713_271329

theorem power_nine_mod_seven : 9^123 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_mod_seven_l2713_271329


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2713_271321

theorem right_rectangular_prism_volume
  (face_area1 face_area2 face_area3 : ℝ)
  (h1 : face_area1 = 6.5)
  (h2 : face_area2 = 8)
  (h3 : face_area3 = 13)
  : ∃ (l w h : ℝ),
    l * w = face_area1 ∧
    w * h = face_area2 ∧
    l * h = face_area3 ∧
    l * w * h = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2713_271321


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l2713_271326

/-- Converts Australian dollars to US dollars given an exchange rate -/
def aud_to_usd (aud_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  aud_amount * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_aud = 45000000)
  (h2 : exchange_rate = 0.7) :
  aud_to_usd damage_aud exchange_rate = 31500000 :=
by sorry

end NUMINAMATH_CALUDE_flood_damage_conversion_l2713_271326


namespace NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l2713_271317

theorem max_cars_ac_no_stripes (total_cars : Nat) (cars_no_ac : Nat) (cars_with_stripes : Nat)
  (red_cars : Nat) (red_cars_ac_stripes : Nat) (cars_2000s : Nat) (cars_2010s : Nat)
  (min_new_cars_stripes : Nat) (h1 : total_cars = 150) (h2 : cars_no_ac = 47)
  (h3 : cars_with_stripes = 65) (h4 : red_cars = 25) (h5 : red_cars_ac_stripes = 10)
  (h6 : cars_2000s = 30) (h7 : cars_2010s = 43) (h8 : min_new_cars_stripes = 39)
  (h9 : min_new_cars_stripes ≤ cars_2000s + cars_2010s) :
  (cars_2000s + cars_2010s) - min_new_cars_stripes - red_cars_ac_stripes = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l2713_271317


namespace NUMINAMATH_CALUDE_parallelogram_other_vertices_y_sum_l2713_271307

/-- A parallelogram with two opposite vertices at (2,15) and (8,-2) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (2, 15)
  v2 : ℝ × ℝ := (8, -2)
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  is_parallelogram : True  -- We assume this is a valid parallelogram

/-- The sum of y-coordinates of the other two vertices is 13 -/
theorem parallelogram_other_vertices_y_sum (p : Parallelogram) : 
  (p.v3).2 + (p.v4).2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_other_vertices_y_sum_l2713_271307


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_set_l2713_271325

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) (h1 : a > 1) :
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x a ≥ m) → a = 7 :=
sorry

-- Theorem for part 2
theorem inequality_solution_set (x : ℝ) :
  f x 7 ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_set_l2713_271325


namespace NUMINAMATH_CALUDE_trash_can_prices_min_A_type_cans_l2713_271365

-- Define variables for trash can prices
variable (price_A : ℝ) (price_B : ℝ)

-- Define the cost equations
def cost_equation_1 (price_A price_B : ℝ) : Prop :=
  3 * price_A + 4 * price_B = 580

def cost_equation_2 (price_A price_B : ℝ) : Prop :=
  6 * price_A + 5 * price_B = 860

-- Define the total number of trash cans
def total_cans : ℕ := 200

-- Define the budget constraint
def budget : ℝ := 15000

-- Theorem for part 1
theorem trash_can_prices :
  cost_equation_1 price_A price_B ∧ cost_equation_2 price_A price_B →
  price_A = 60 ∧ price_B = 100 := by sorry

-- Theorem for part 2
theorem min_A_type_cans (num_A : ℕ) :
  num_A * price_A + (total_cans - num_A) * price_B ≤ budget →
  num_A ≥ 125 := by sorry

end NUMINAMATH_CALUDE_trash_can_prices_min_A_type_cans_l2713_271365
