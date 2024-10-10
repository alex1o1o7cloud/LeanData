import Mathlib

namespace apples_for_juice_l1585_158567

/-- Given that 36 apples make 27 liters of apple juice, prove that 12 apples make 9 liters of apple juice -/
theorem apples_for_juice (apples : ℕ) (juice : ℕ) (h : 36 * juice = 27 * apples) : 
  12 * juice = 9 * apples :=
by sorry

end apples_for_juice_l1585_158567


namespace g_domain_is_correct_l1585_158565

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-6) 9

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of g
def g_domain : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem g_domain_is_correct : 
  {x : ℝ | g x ∈ f_domain} = g_domain := by sorry

end g_domain_is_correct_l1585_158565


namespace right_triangle_vector_property_l1585_158520

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the theorem
theorem right_triangle_vector_property (t : RightTriangleABC) (x : ℝ) 
  (h1 : t.C.1 - t.A.1 = 2 ∧ t.C.2 - t.A.2 = 4)
  (h2 : t.C.1 - t.B.1 = -6 ∧ t.C.2 - t.B.2 = x) :
  x = 3 := by
  sorry

-- The proof is omitted as per instructions

end right_triangle_vector_property_l1585_158520


namespace parabola_reflects_to_parallel_l1585_158534

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a curve in 2D space -/
def CurveEquation : Type := Point → Prop

/-- The equation of a parabola y^2 = 2Cx + C^2 -/
def ParabolaEquation (C : ℝ) : CurveEquation :=
  fun p => p.y^2 = 2*C*p.x + C^2

/-- A ray of light -/
structure Ray where
  origin : Point
  direction : Point

/-- The reflection of a ray off a curve at a point -/
def ReflectedRay (curve : CurveEquation) (incidentRay : Ray) (reflectionPoint : Point) : Ray :=
  sorry

/-- The theorem stating that a parabola reflects rays from the origin into parallel rays -/
theorem parabola_reflects_to_parallel (C : ℝ) :
  ∀ (p : Point), ParabolaEquation C p →
  ∀ (incidentRay : Ray),
    incidentRay.origin = ⟨0, 0⟩ →
    (ReflectedRay (ParabolaEquation C) incidentRay p).direction.y = 0 :=
  sorry

end parabola_reflects_to_parallel_l1585_158534


namespace sufficient_but_not_necessary_l1585_158552

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary (s : Sequence) 
  (h_pos : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
sorry

end sufficient_but_not_necessary_l1585_158552


namespace arithmetic_geometric_sum_sixth_term_l1585_158528

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop := sorry

/-- Sum of the first n terms of a sequence -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sum_sixth_term 
  (a : ℕ → ℝ) 
  (h_ag : arithmetic_geometric_sequence a)
  (h_s2 : S a 2 = 1)
  (h_s4 : S a 4 = 3) : 
  S a 6 = 7 := by sorry

end arithmetic_geometric_sum_sixth_term_l1585_158528


namespace lucky_sum_equality_l1585_158569

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to form a sum of s using k distinct natural numbers less than n -/
def sumCombinations (s k n : ℕ) : ℕ := sorry

/-- The probability of event A: "the lucky sum in the main draw is 63" -/
def probA (n : ℕ) : ℚ :=
  (sumCombinations 63 10 n : ℚ) / choose n 10

/-- The probability of event B: "the lucky sum in the additional draw is 44" -/
def probB (n : ℕ) : ℚ :=
  (sumCombinations 44 8 n : ℚ) / choose n 8

theorem lucky_sum_equality :
  ∀ n : ℕ, (n ≥ 10 ∧ probA n = probB n) ↔ n = 18 := by sorry

end lucky_sum_equality_l1585_158569


namespace percentage_commutation_l1585_158542

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 24) : 0.4 * (0.3 * n) = 24 := by
  sorry

end percentage_commutation_l1585_158542


namespace minjin_apples_l1585_158570

theorem minjin_apples : ∃ (initial : ℕ), 
  (initial % 8 = 0) ∧ 
  (6 * ((initial / 8) + 8 - 30) = 12) ∧ 
  (initial = 192) := by
  sorry

end minjin_apples_l1585_158570


namespace rd_investment_exceeds_200_million_in_2019_l1585_158577

/-- Proves that 2019 is the first year when the annual R&D bonus investment exceeds $200 million -/
theorem rd_investment_exceeds_200_million_in_2019 
  (initial_investment : ℝ) 
  (annual_increase_rate : ℝ) 
  (h1 : initial_investment = 130) 
  (h2 : annual_increase_rate = 0.12) : 
  ∃ (n : ℕ), 
    (n = 2019) ∧ 
    (initial_investment * (1 + annual_increase_rate) ^ (n - 2015) > 200) ∧ 
    (∀ m : ℕ, m < n → initial_investment * (1 + annual_increase_rate) ^ (m - 2015) ≤ 200) := by
  sorry

end rd_investment_exceeds_200_million_in_2019_l1585_158577


namespace necessary_but_not_sufficient_l1585_158572

theorem necessary_but_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x ≤ 1/2 ∧ y ≤ 1/2 → x + y ≤ 1) ∧
  (∃ x y : ℝ, x + y ≤ 1 ∧ ¬(x ≤ 1/2 ∧ y ≤ 1/2)) :=
by sorry

end necessary_but_not_sufficient_l1585_158572


namespace betty_balance_l1585_158578

/-- Betty's account balance given Gina's account information -/
theorem betty_balance (gina_account1 gina_account2 betty_balance : ℚ) : 
  gina_account1 = (1 / 4 : ℚ) * betty_balance →
  gina_account2 = (1 / 4 : ℚ) * betty_balance →
  gina_account1 + gina_account2 = 1728 →
  betty_balance = 3456 := by
  sorry

end betty_balance_l1585_158578


namespace rational_roots_of_polynomial_l1585_158573

theorem rational_roots_of_polynomial (x : ℚ) :
  (4 * x^4 - 3 * x^3 - 13 * x^2 + 5 * x + 2 = 0) ↔ (x = 2 ∨ x = -1/4) :=
by sorry

end rational_roots_of_polynomial_l1585_158573


namespace circle_reflection_translation_l1585_158538

def reflect_across_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

def translate_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_up (reflect_across_y_axis center.1 center.2) 5) = (-3, 1) := by
  sorry

end circle_reflection_translation_l1585_158538


namespace solution_set_and_range_l1585_158587

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ x ∈ Set.Icc (-1) 2) ∧
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, f x < |a - 2|) ↔ a > 6) := by sorry

end solution_set_and_range_l1585_158587


namespace production_volume_equation_l1585_158506

theorem production_volume_equation (x : ℝ) : 
  (200 : ℝ) + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 ↔ 
  (∃ y : ℝ, y > 0 ∧ 
    200 * (1 + y + (1 + y)^2) = 1400 ∧
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 3 → 
      (200 * (1 + y)^(n - 1) = 200 * (1 + x)^(n - 1)))) :=
by sorry

end production_volume_equation_l1585_158506


namespace sue_necklace_purple_beads_l1585_158592

theorem sue_necklace_purple_beads :
  ∀ (purple blue green : ℕ),
    purple + blue + green = 46 →
    blue = 2 * purple →
    green = blue + 11 →
    purple = 7 := by
  sorry

end sue_necklace_purple_beads_l1585_158592


namespace melanie_missed_games_l1585_158597

theorem melanie_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 64)
  (h2 : attended_games = 32) :
  total_games - attended_games = 32 := by
  sorry

end melanie_missed_games_l1585_158597


namespace pencils_per_box_l1585_158589

/-- The number of pencils Louise has for each color and the number of boxes --/
structure PencilData where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  boxes : ℕ

/-- The conditions of Louise's pencil organization --/
def validPencilData (d : PencilData) : Prop :=
  d.red = 20 ∧
  d.blue = 2 * d.red ∧
  d.yellow = 40 ∧
  d.green = d.red + d.blue ∧
  d.boxes = 8

/-- The theorem stating that each box holds 20 pencils --/
theorem pencils_per_box (d : PencilData) (h : validPencilData d) :
  (d.red + d.blue + d.yellow + d.green) / d.boxes = 20 := by
  sorry

#check pencils_per_box

end pencils_per_box_l1585_158589


namespace sum_is_composite_l1585_158541

theorem sum_is_composite (a b : ℤ) (h : 56 * a = 65 * b) : 
  ∃ (x y : ℤ), x > 1 ∧ y > 1 ∧ a + b = x * y := by
sorry

end sum_is_composite_l1585_158541


namespace julio_mocktail_lime_juice_l1585_158502

/-- Proves that Julio uses 1 tablespoon of lime juice per mocktail -/
theorem julio_mocktail_lime_juice :
  -- Define the problem parameters
  let days : ℕ := 30
  let mocktails_per_day : ℕ := 1
  let lime_juice_per_lime : ℚ := 2
  let limes_per_dollar : ℚ := 3
  let total_spent : ℚ := 5

  -- Calculate the total number of limes bought
  let total_limes : ℚ := total_spent * limes_per_dollar

  -- Calculate the total amount of lime juice
  let total_lime_juice : ℚ := total_limes * lime_juice_per_lime

  -- Calculate the amount of lime juice per mocktail
  let lime_juice_per_mocktail : ℚ := total_lime_juice / (days * mocktails_per_day)

  -- Prove that the amount of lime juice per mocktail is 1 tablespoon
  lime_juice_per_mocktail = 1 := by sorry

end julio_mocktail_lime_juice_l1585_158502


namespace value_of_d_l1585_158594

theorem value_of_d (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 4 * d) : d = 15 / 8 := by
  sorry

end value_of_d_l1585_158594


namespace kevin_food_expenditure_l1585_158553

theorem kevin_food_expenditure (total_budget : ℕ) (samuel_ticket : ℕ) (samuel_food_drinks : ℕ) (kevin_drinks : ℕ) :
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_food_drinks = 6 →
  kevin_drinks = 2 →
  total_budget = samuel_ticket + samuel_food_drinks →
  ∃ (kevin_food : ℕ), total_budget = samuel_ticket + kevin_drinks + kevin_food ∧ kevin_food = 4 :=
by sorry

end kevin_food_expenditure_l1585_158553


namespace intersection_of_lines_l1585_158529

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ∧ 
  (x = 75 / 23) ∧ (y = 161 / 23) := by
  sorry

end intersection_of_lines_l1585_158529


namespace sphere_volume_for_cube_surface_l1585_158547

theorem sphere_volume_for_cube_surface (cube_side : ℝ) (L : ℝ) : 
  cube_side = 3 →
  (4 / 3 * π * (((6 * cube_side^2) / (4 * π))^(3/2))) = L * Real.sqrt 15 / Real.sqrt π →
  L = 84 := by
  sorry

end sphere_volume_for_cube_surface_l1585_158547


namespace hyper_box_side_sum_l1585_158559

/-- The sum of side lengths of a four-dimensional rectangular hyper-box with given face volumes -/
theorem hyper_box_side_sum (W X Y Z : ℝ) 
  (h1 : W * X * Y = 60)
  (h2 : W * X * Z = 80)
  (h3 : W * Y * Z = 120)
  (h4 : X * Y * Z = 60) :
  W + X + Y + Z = 318.5 := by
  sorry

end hyper_box_side_sum_l1585_158559


namespace intersection_implies_equality_l1585_158543

theorem intersection_implies_equality (k b a c : ℝ) : 
  k ≠ b → 
  (∃! p : ℝ × ℝ, (p.2 = k * p.1 + k) ∧ (p.2 = b * p.1 + b) ∧ (p.2 = a * p.1 + c)) →
  a = c := by
sorry

end intersection_implies_equality_l1585_158543


namespace tangent_line_to_exp_curve_l1585_158583

theorem tangent_line_to_exp_curve (x y : ℝ) :
  (∃ (m b : ℝ), y = m * x + b ∧ 
    (∀ (x₀ : ℝ), Real.exp x₀ = m * x₀ + b → x₀ = 1 ∨ x₀ = x) ∧
    0 = m * 1 + b) →
  Real.exp 2 * x - y - Real.exp 2 = 0 :=
by sorry

end tangent_line_to_exp_curve_l1585_158583


namespace triangle_angle_zero_l1585_158568

theorem triangle_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 := by sorry

end triangle_angle_zero_l1585_158568


namespace polynomial_division_remainder_l1585_158554

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 + 2*x^5 - 3*x^4 + x^3 - 2*x^2 + 5*x - 1 =
  (x - 1) * (x + 2) * (x - 3) * q + (17*x^2 - 52*x + 38) := by
  sorry

end polynomial_division_remainder_l1585_158554


namespace perpendicular_vectors_m_value_l1585_158588

/-- Given two vectors a and b in ℝ², prove that if a = (1,2) and b = (-1,m) are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) : 
  a = (1, 2) → b = (-1, m) → a.1 * b.1 + a.2 * b.2 = 0 → m = 1/2 := by
  sorry

end perpendicular_vectors_m_value_l1585_158588


namespace cyclic_sum_inequality_l1585_158550

theorem cyclic_sum_inequality (x y z : ℝ) : 
  let a := x + y + z
  ((a - x)^4 + (a - y)^4 + (a - z)^4) + 
  2 * (x^3*y + x^3*z + y^3*x + y^3*z + z^3*x + z^3*y) + 
  4 * (x^2*y^2 + y^2*z^2 + z^2*x^2) + 
  8 * x*y*z*a ≥ 
  ((a - x)^2*(a^2 - x^2) + (a - y)^2*(a^2 - y^2) + (a - z)^2*(a^2 - z^2)) := by
sorry

end cyclic_sum_inequality_l1585_158550


namespace candy_probability_difference_l1585_158544

theorem candy_probability_difference : 
  let total_candies : ℕ := 2004
  let banana_candies : ℕ := 1002
  let apple_candies : ℕ := 1002
  let different_flavor_prob : ℚ := banana_candies * apple_candies / (total_candies * (total_candies - 1))
  let same_flavor_prob : ℚ := (banana_candies * (banana_candies - 1) + apple_candies * (apple_candies - 1)) / (total_candies * (total_candies - 1))
  different_flavor_prob - same_flavor_prob = 1 / 2003 := by
sorry

end candy_probability_difference_l1585_158544


namespace share_of_b_l1585_158560

theorem share_of_b (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  a + b + c = 645 → 
  b = 134 := by
sorry

end share_of_b_l1585_158560


namespace milk_cost_l1585_158574

/-- If 4 boxes of milk cost 26 yuan, then 6 boxes of the same milk will cost 39 yuan. -/
theorem milk_cost (cost : ℕ) (boxes : ℕ) (h1 : cost = 26) (h2 : boxes = 4) :
  (cost / boxes) * 6 = 39 :=
by sorry

end milk_cost_l1585_158574


namespace solution_problem_l1585_158510

theorem solution_problem (a₁ a₂ a₃ a₄ a₅ b : ℤ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
                a₄ ≠ a₅)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 9)
  (h_root : (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009) :
  b = 10 := by
sorry

end solution_problem_l1585_158510


namespace only_first_equation_has_nonzero_solution_l1585_158576

theorem only_first_equation_has_nonzero_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (a^2 + b^2) = a ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = Real.sqrt a * Real.sqrt b → a = 0 ∧ b = 0) ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = a * b → a = 0 ∧ b = 0) := by
  sorry

end only_first_equation_has_nonzero_solution_l1585_158576


namespace quadratic_equation_roots_l1585_158599

theorem quadratic_equation_roots (k : ℤ) :
  let f := fun x : ℝ => k * x^2 - (4*k + 1) * x + 3*k + 3
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁^2 + x₂^2 = (3 * Real.sqrt 5 / 2)^2 → k = 2) :=
by sorry

end quadratic_equation_roots_l1585_158599


namespace inequality_order_l1585_158557

theorem inequality_order (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry

end inequality_order_l1585_158557


namespace ellipse_parameter_sum_l1585_158504

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to both foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_parameters (e : Ellipse) : EllipseParameters :=
  sorry

/-- The main theorem: sum of center coordinates and axes lengths for the given ellipse -/
theorem ellipse_parameter_sum (e : Ellipse) 
    (h : e.F₁ = (0, 2) ∧ e.F₂ = (6, 2) ∧ e.sum_distances = 10) : 
    let p := compute_ellipse_parameters e
    p.h + p.k + p.a + p.b = 14 :=
  sorry

end ellipse_parameter_sum_l1585_158504


namespace angle_measure_proof_l1585_158500

theorem angle_measure_proof : 
  ∀ x : ℝ, 
    (90 - x = (1/7) * x + 26) → 
    x = 56 := by
  sorry

end angle_measure_proof_l1585_158500


namespace yellow_ball_players_l1585_158598

theorem yellow_ball_players (total : ℕ) (white : ℕ) (both : ℕ) (yellow : ℕ) : 
  total = 35 → white = 26 → both = 19 → yellow = 28 → 
  total = white + yellow - both :=
by sorry

end yellow_ball_players_l1585_158598


namespace count_less_than_10000_l1585_158527

def count_numbers_with_at_most_three_digits (n : ℕ) : ℕ :=
  sorry

theorem count_less_than_10000 : 
  count_numbers_with_at_most_three_digits 10000 = 3231 := by
  sorry

end count_less_than_10000_l1585_158527


namespace inequality_proof_l1585_158595

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := by
  sorry

end inequality_proof_l1585_158595


namespace max_value_theorem_l1585_158507

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
sorry

end max_value_theorem_l1585_158507


namespace zero_point_in_interval_l1585_158524

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end zero_point_in_interval_l1585_158524


namespace mundane_goblet_points_difference_l1585_158540

def round_robin_tournament (n : ℕ) := n * (n - 1) / 2

theorem mundane_goblet_points_difference :
  let num_teams : ℕ := 6
  let num_matches := round_robin_tournament num_teams
  let max_points := num_matches * 3
  let min_points := num_matches * 2
  max_points - min_points = 15 := by
  sorry

end mundane_goblet_points_difference_l1585_158540


namespace movie_production_cost_l1585_158533

def opening_weekend_revenue : ℝ := 120000000
def total_revenue_multiplier : ℝ := 3.5
def production_company_share : ℝ := 0.60
def profit : ℝ := 192000000

theorem movie_production_cost :
  let total_revenue := opening_weekend_revenue * total_revenue_multiplier
  let production_company_revenue := total_revenue * production_company_share
  let production_cost := production_company_revenue - profit
  production_cost = 60000000 := by sorry

end movie_production_cost_l1585_158533


namespace unique_prime_triplet_l1585_158516

theorem unique_prime_triplet :
  ∀ a b c : ℕ+,
    (Nat.Prime (a + b * c) ∧ 
     Nat.Prime (b + a * c) ∧ 
     Nat.Prime (c + a * b)) ∧
    ((a + b * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((b + a * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((c + a * b) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) →
    a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end unique_prime_triplet_l1585_158516


namespace aluminium_count_l1585_158549

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := sorry

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 78

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Theorem stating that the number of Aluminium atoms in the compound is 1 -/
theorem aluminium_count : n = 1 := by sorry

end aluminium_count_l1585_158549


namespace equation_is_ellipse_l1585_158561

def equation (x y : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x - 8*y + 9 = 0

def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (x y : ℝ), f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end equation_is_ellipse_l1585_158561


namespace g_max_value_l1585_158586

/-- The function g(x) = 4x - x^3 -/
def g (x : ℝ) : ℝ := 4 * x - x^3

/-- The maximum value of g(x) on [0, 2] is 8√3/9 -/
theorem g_max_value : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = (8 * Real.sqrt 3) / 9 := by
sorry

end g_max_value_l1585_158586


namespace mass_percentage_iodine_value_of_x_l1585_158564

-- Define constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_I : ℝ := 126.90
def molar_mass_H2O : ℝ := 18.015

-- Define the sample mass
def sample_mass : ℝ := 50

-- Define variables for masses of AlI₃ and H₂O in the sample
variable (mass_AlI3 : ℝ)
variable (mass_H2O : ℝ)

-- Calculate molar mass of AlI₃
def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

-- Define the theorem for mass percentage of iodine
theorem mass_percentage_iodine :
  let mass_iodine := mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3)
  (mass_iodine / sample_mass) * 100 = 
  (mass_AlI3 * (3 * molar_mass_I / molar_mass_AlI3) / sample_mass) * 100 :=
by sorry

-- Define the theorem for the value of x
theorem value_of_x :
  let moles_water := mass_H2O / molar_mass_H2O
  let moles_AlI3 := mass_AlI3 / molar_mass_AlI3
  (moles_water / moles_AlI3) = 
  (mass_H2O / molar_mass_H2O) / (mass_AlI3 / molar_mass_AlI3) :=
by sorry

end mass_percentage_iodine_value_of_x_l1585_158564


namespace chord_division_ratio_l1585_158579

/-- Given a circle with radius 11 and a chord of length 18 intersected by a diameter at a point 7 units from the center, 
    the point of intersection divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
  (h_r : r = 11) (h_chord : chord_length = 18) (h_dist : intersection_distance = 7) :
  ∃ (x y : ℝ), (x + y = chord_length ∧ 
    ((x / y = 2 ∧ y / x = 1/2) ∨ (x / y = 1/2 ∧ y / x = 2)) ∧
    x * y = (r - intersection_distance) * (r + intersection_distance)) :=
sorry

end chord_division_ratio_l1585_158579


namespace arithmetic_sequence_with_geometric_mean_l1585_158545

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_with_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1) (h2 : d ≠ 0)
  (h3 : arithmetic_sequence a d)
  (h4 : a 2 ^ 2 = a 1 * a 4) :
  d = 1 := by sorry

end arithmetic_sequence_with_geometric_mean_l1585_158545


namespace fraction_equation_solution_l1585_158513

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 2) / (x - 3) = (x - 4) / (x + 5) :=
by
  -- The unique solution is x = 1/7
  use 1/7
  sorry

end fraction_equation_solution_l1585_158513


namespace carrot_picking_l1585_158522

theorem carrot_picking (carol_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7) :
  good_carrots + bad_carrots - carol_carrots = 16 := by
  sorry

end carrot_picking_l1585_158522


namespace vertex_on_x_axis_l1585_158501

/-- A quadratic function f(x) = x^2 + 2x + k has its vertex on the x-axis if and only if k = 1 -/
theorem vertex_on_x_axis (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k ≥ x^2 + 2*x + k) ↔ 
  k = 1 := by
sorry

end vertex_on_x_axis_l1585_158501


namespace triangle_inequality_l1585_158525

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 2) : 
  a^2 + b^2 + c^2 + 2*a*b*c < 2 := by
sorry

end triangle_inequality_l1585_158525


namespace C_symmetric_origin_C_area_greater_than_pi_l1585_158535

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^4 + p.2^2 = 1}

-- Symmetry with respect to the origin
theorem C_symmetric_origin : ∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ C := by sorry

-- Area enclosed by C is greater than π
theorem C_area_greater_than_pi : ∃ (A : ℝ), A > π ∧ (∀ (x y : ℝ), (x, y) ∈ C → x^2 + y^2 ≤ A) := by sorry

end C_symmetric_origin_C_area_greater_than_pi_l1585_158535


namespace lee_science_class_l1585_158505

theorem lee_science_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) : 
  total = 56 → girls_ratio = 4 → boys_ratio = 3 → 
  (girls_ratio + boys_ratio) * (total / (girls_ratio + boys_ratio)) * boys_ratio / girls_ratio = 24 := by
sorry

end lee_science_class_l1585_158505


namespace arithmetic_sequence_common_difference_l1585_158517

/-- An arithmetic sequence with first term 5 and the sum of the 6th and 8th terms equal to 58 has a common difference of 4. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 5 →
  a 6 + a 8 = 58 →
  a 2 - a 1 = 4 := by
sorry

end arithmetic_sequence_common_difference_l1585_158517


namespace next_divisible_by_sum_of_digits_l1585_158537

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- The next number after 1232 that is divisible by the sum of its digits -/
theorem next_divisible_by_sum_of_digits :
  ∃ (n : ℕ), n > 1232 ∧
    isDivisibleBySumOfDigits n ∧
    ∀ (m : ℕ), 1232 < m ∧ m < n → ¬isDivisibleBySumOfDigits m :=
by sorry

end next_divisible_by_sum_of_digits_l1585_158537


namespace product_equality_l1585_158508

theorem product_equality (a : ℝ) (h : a ≠ 0 ∧ a ≠ 2 ∧ a ≠ -2) :
  (a^2 + 2*a + 4 + 8/a + 16/a^2 + 64/((a-2)*a^2)) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2 - 64/((a+2)*a^2))
  =
  (a^2 + 2*a + 4 + 8/a + 16/a^2) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2) :=
by sorry

end product_equality_l1585_158508


namespace special_sequence_1000th_term_l1585_158536

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2007 ∧ 
  a 2 = 2008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

/-- The 1000th term of the special sequence is 2340 -/
theorem special_sequence_1000th_term (a : ℕ → ℕ) (h : SpecialSequence a) : 
  a 1000 = 2340 := by
  sorry

end special_sequence_1000th_term_l1585_158536


namespace symmetry_x_axis_l1585_158523

/-- Given two points P and Q in the Cartesian coordinate system,
    prove that if P is symmetric to Q with respect to the x-axis,
    then the sum of their x-coordinates minus 3 and the negation of Q's y-coordinate minus 1
    is equal to 3. -/
theorem symmetry_x_axis (a b : ℝ) :
  let P : ℝ × ℝ := (a - 3, 1)
  let Q : ℝ × ℝ := (2, b + 1)
  (P.1 = Q.1) →  -- x-coordinates are equal
  (P.2 = -Q.2) → -- y-coordinates are opposite
  a + b = 3 := by
sorry

end symmetry_x_axis_l1585_158523


namespace product_sum_theorem_l1585_158548

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a + b + c = 21) : 
  a*b + b*c + a*c = 100 := by
sorry

end product_sum_theorem_l1585_158548


namespace only_positive_number_l1585_158562

theorem only_positive_number (numbers : Set ℝ) : 
  numbers = {0, 5, -1/2, -Real.sqrt 2} → 
  (∃ x ∈ numbers, x > 0) ∧ (∀ y ∈ numbers, y > 0 → y = 5) := by
sorry

end only_positive_number_l1585_158562


namespace algebraic_sum_equals_one_l1585_158519

theorem algebraic_sum_equals_one (a b c x : ℝ) 
  (ha : a + x^2 = 2006)
  (hb : b + x^2 = 2007)
  (hc : c + x^2 = 2008)
  (habc : a * b * c = 3) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 :=
by sorry

end algebraic_sum_equals_one_l1585_158519


namespace vector_addition_result_l1585_158514

theorem vector_addition_result (a b : ℝ × ℝ) :
  a = (2, 1) → b = (1, 5) → 2 • a + b = (5, 7) := by
  sorry

end vector_addition_result_l1585_158514


namespace not_all_zero_iff_one_nonzero_l1585_158575

theorem not_all_zero_iff_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end not_all_zero_iff_one_nonzero_l1585_158575


namespace find_number_l1585_158512

theorem find_number (n : ℝ) : (0.47 * 1442 - 0.36 * n) + 63 = 3 → n = 2049.28 := by
  sorry

end find_number_l1585_158512


namespace three_by_four_grid_squares_l1585_158563

/-- A structure representing a grid of squares -/
structure SquareGrid where
  rows : Nat
  cols : Nat
  total_small_squares : Nat

/-- Function to count the total number of squares in a grid -/
def count_total_squares (grid : SquareGrid) : Nat :=
  sorry

/-- Theorem stating that a 3x4 grid of 12 small squares contains 17 total squares -/
theorem three_by_four_grid_squares :
  let grid := SquareGrid.mk 3 4 12
  count_total_squares grid = 17 :=
by sorry

end three_by_four_grid_squares_l1585_158563


namespace regular_21gon_symmetry_sum_l1585_158584

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 21-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is equal to 38 -/
theorem regular_21gon_symmetry_sum :
  (L n : ℚ) + R n = 38 := by sorry

end regular_21gon_symmetry_sum_l1585_158584


namespace total_cost_nine_knives_l1585_158546

/-- Calculates the total cost of sharpening knives based on a specific pricing structure. -/
def total_sharpening_cost (num_knives : ℕ) : ℚ :=
  let first_knife_cost : ℚ := 5
  let next_three_cost : ℚ := 4
  let remaining_cost : ℚ := 3
  let num_next_three : ℕ := min 3 (num_knives - 1)
  let num_remaining : ℕ := max 0 (num_knives - 4)
  first_knife_cost + 
  (next_three_cost * num_next_three) + 
  (remaining_cost * num_remaining)

/-- Theorem stating that the total cost to sharpen 9 knives is $32.00. -/
theorem total_cost_nine_knives : 
  total_sharpening_cost 9 = 32 := by
  sorry

end total_cost_nine_knives_l1585_158546


namespace remaining_average_l1585_158593

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 6 →
  subset = 4 →
  total_avg = 8 →
  subset_avg = 5 →
  (total_avg * total - subset_avg * subset) / (total - subset) = 14 := by
  sorry

end remaining_average_l1585_158593


namespace perpendicular_planes_l1585_158509

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_perp_α : perpendicular m α) 
  (h_m_para_β : parallel m β) : 
  plane_perpendicular α β :=
sorry

end perpendicular_planes_l1585_158509


namespace cards_in_unfilled_box_l1585_158581

theorem cards_in_unfilled_box (total_cards : ℕ) (cards_per_box : ℕ) 
  (h1 : total_cards = 94) (h2 : cards_per_box = 8) : 
  total_cards % cards_per_box = 6 := by
  sorry

end cards_in_unfilled_box_l1585_158581


namespace mr_green_potato_yield_l1585_158532

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield from Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry


end mr_green_potato_yield_l1585_158532


namespace product_of_squares_and_prime_l1585_158515

theorem product_of_squares_and_prime : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end product_of_squares_and_prime_l1585_158515


namespace tangency_condition_l1585_158555

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m*(y+1)^2 = 1

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y m ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' m → (x', y') = (x, y)

-- State the theorem
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ m = 2 := by sorry

end tangency_condition_l1585_158555


namespace negation_equivalence_l1585_158521

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end negation_equivalence_l1585_158521


namespace total_winter_clothing_l1585_158539

/-- The number of boxes of winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of winter clothing pieces is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
sorry

end total_winter_clothing_l1585_158539


namespace fraction_problem_l1585_158518

theorem fraction_problem (N : ℝ) (x y : ℤ) :
  N = 30 →
  0.5 * N = (x / y : ℝ) * N + 10 →
  (x / y : ℝ) = 1 / 6 := by
  sorry

end fraction_problem_l1585_158518


namespace inequality_proof_l1585_158558

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
  (sum_condition : a + b + c = 2) : 
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 27 / 13) ∧ 
  ((1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) = 27 / 13) ↔ 
   (a = 2/3 ∧ b = 2/3 ∧ c = 2/3)) :=
by sorry

end inequality_proof_l1585_158558


namespace unique_n_with_divisor_property_l1585_158582

def has_ten_divisors (n : ℕ) : Prop :=
  ∃ (d : Fin 10 → ℕ), d 0 = 1 ∧ d 9 = n ∧
    (∀ i : Fin 9, d i < d (i + 1)) ∧
    (∀ m : ℕ, m ∣ n ↔ ∃ i : Fin 10, d i = m)

theorem unique_n_with_divisor_property :
  ∀ n : ℕ, n > 0 →
    has_ten_divisors n →
    (∃ (d : Fin 10 → ℕ), 2 * n = (d 4)^2 + (d 5)^2 - 1) →
    n = 272 :=
sorry

end unique_n_with_divisor_property_l1585_158582


namespace farm_animals_ratio_l1585_158596

theorem farm_animals_ratio :
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := goats - 33
  (pigs : ℚ) / ducks = 1 / 3 :=
by sorry

end farm_animals_ratio_l1585_158596


namespace ethanol_in_full_tank_l1585_158571

def tank_capacity : ℝ := 212
def fuel_A_volume : ℝ := 98
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_B_ethanol_percentage : ℝ := 0.16

theorem ethanol_in_full_tank : 
  let fuel_B_volume := tank_capacity - fuel_A_volume
  let ethanol_in_A := fuel_A_volume * fuel_A_ethanol_percentage
  let ethanol_in_B := fuel_B_volume * fuel_B_ethanol_percentage
  ethanol_in_A + ethanol_in_B = 30 := by
sorry

end ethanol_in_full_tank_l1585_158571


namespace jaydon_rachel_ratio_l1585_158503

-- Define the number of cans for each person
def mark_cans : ℕ := 100
def jaydon_cans : ℕ := 25
def rachel_cans : ℕ := 10

-- Define the total number of cans
def total_cans : ℕ := 135

-- Define the conditions
axiom mark_jaydon_relation : mark_cans = 4 * jaydon_cans
axiom total_cans_sum : total_cans = mark_cans + jaydon_cans + rachel_cans
axiom jaydon_rachel_relation : ∃ k : ℕ, jaydon_cans = k * rachel_cans + 5

-- Theorem to prove
theorem jaydon_rachel_ratio : 
  (jaydon_cans : ℚ) / rachel_cans = 5 / 2 := by sorry

end jaydon_rachel_ratio_l1585_158503


namespace stock_change_theorem_l1585_158531

theorem stock_change_theorem (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.15)
  let day2_value := day1_value * (1 + 0.25)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 6.25 := by
  sorry

end stock_change_theorem_l1585_158531


namespace line_plane_perpendicularity_l1585_158580

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n) 
  (h_perp : perpendicular m α) 
  (h_contained : contained_in n α) : 
  perpendicular_lines m n :=
sorry

end line_plane_perpendicularity_l1585_158580


namespace step_waddle_difference_is_six_l1585_158526

/-- The number of steps Gerald takes between consecutive lamp posts -/
def gerald_steps : ℕ := 55

/-- The number of waddles Patricia takes between consecutive lamp posts -/
def patricia_waddles : ℕ := 15

/-- The number of lamp posts -/
def num_posts : ℕ := 31

/-- The total distance between the first and last lamp post in feet -/
def total_distance : ℕ := 3720

/-- Gerald's step length in feet -/
def gerald_step_length : ℚ := total_distance / (gerald_steps * (num_posts - 1))

/-- Patricia's waddle length in feet -/
def patricia_waddle_length : ℚ := total_distance / (patricia_waddles * (num_posts - 1))

/-- The difference between Gerald's step length and Patricia's waddle length -/
def step_waddle_difference : ℚ := patricia_waddle_length - gerald_step_length

theorem step_waddle_difference_is_six :
  step_waddle_difference = 6 := by sorry

end step_waddle_difference_is_six_l1585_158526


namespace chef_initial_potatoes_l1585_158551

/-- Represents the number of fries that can be made from one potato -/
def fries_per_potato : ℕ := 25

/-- Represents the total number of fries needed -/
def total_fries_needed : ℕ := 200

/-- Represents the number of potatoes leftover after making the required fries -/
def leftover_potatoes : ℕ := 7

/-- Calculates the initial number of potatoes the chef had -/
def initial_potatoes : ℕ := (total_fries_needed / fries_per_potato) + leftover_potatoes

/-- Proves that the initial number of potatoes is 15 -/
theorem chef_initial_potatoes :
  initial_potatoes = 15 :=
by sorry

end chef_initial_potatoes_l1585_158551


namespace cosine_vertical_shift_l1585_158511

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_oscillation : ∀ x : ℝ, 0 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 4) : 
  d = 2 := by
sorry

end cosine_vertical_shift_l1585_158511


namespace gmat_exam_problem_l1585_158585

theorem gmat_exam_problem (total : ℕ) (h_total : total > 0) :
  let first_correct := (80 : ℚ) / 100 * total
  let second_correct := (75 : ℚ) / 100 * total
  let neither_correct := (5 : ℚ) / 100 * total
  let both_correct := first_correct + second_correct - total + neither_correct
  (both_correct / total) = (60 : ℚ) / 100 := by
sorry

end gmat_exam_problem_l1585_158585


namespace max_boxes_in_wooden_box_l1585_158590

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions :=
    { length := metersToCentimeters 8
      width := metersToCentimeters 7
      height := metersToCentimeters 6 }
  let smallBox : BoxDimensions :=
    { length := 8
      width := 7
      height := 6 }
  (boxVolume largeBox) / (boxVolume smallBox) = 1000000 := by
  sorry

end max_boxes_in_wooden_box_l1585_158590


namespace quadratic_shift_sum_l1585_158566

/-- Given a quadratic function f(x) = 3x^2 - 2x + 8, when shifted 6 units to the left,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 141 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 8) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 141 := by
  sorry

end quadratic_shift_sum_l1585_158566


namespace exists_counterexample_to_inequality_l1585_158530

theorem exists_counterexample_to_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ (a b c : ℝ), c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2 :=
sorry

end exists_counterexample_to_inequality_l1585_158530


namespace f_g_3_eq_6_l1585_158591

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := x^2 - 8

theorem f_g_3_eq_6 : f (g 3) = 6 := by sorry

end f_g_3_eq_6_l1585_158591


namespace integral_ratio_theorem_l1585_158556

theorem integral_ratio_theorem (a b : ℝ) (h : a < b) :
  let f (x : ℝ) := (1 / 20 + 3 / 10) * x^2
  let g (x : ℝ) := x^2
  (∫ x in a..b, f x) / (∫ x in a..b, g x) = 35 / 100 := by
  sorry

end integral_ratio_theorem_l1585_158556
