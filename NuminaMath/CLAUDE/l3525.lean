import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l3525_352547

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation to be satisfied
def equation (z : ℂ) : Prop :=
  (4 - 3 * i) * z + (6 + 2 * i) = -2 + 15 * i

-- State the theorem
theorem complex_equation_solution :
  equation (-71/7 - 15/7 * i) ∧ i^2 = -1 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3525_352547


namespace NUMINAMATH_CALUDE_hall_area_is_450_l3525_352594

/-- Represents a rectangular hall with specific properties. -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  length_width_diff : length - width = 15

/-- Calculates the area of a rectangular hall. -/
def area (hall : RectangularHall) : ℝ := hall.length * hall.width

/-- Theorem stating that a rectangular hall with the given properties has an area of 450 square units. -/
theorem hall_area_is_450 (hall : RectangularHall) : area hall = 450 := by
  sorry

end NUMINAMATH_CALUDE_hall_area_is_450_l3525_352594


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3525_352509

/-- The number of chairs in the row -/
def n : ℕ := 12

/-- The probability that Mary and James don't sit next to each other -/
def prob_not_adjacent : ℚ := 5/6

/-- The theorem stating the probability of Mary and James not sitting next to each other -/
theorem probability_not_adjacent :
  (1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)) = prob_not_adjacent :=
sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3525_352509


namespace NUMINAMATH_CALUDE_expression_equals_20_times_10_pow_1500_l3525_352517

theorem expression_equals_20_times_10_pow_1500 :
  (2^1500 + 5^1501)^2 - (2^1500 - 5^1501)^2 = 20 * 10^1500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_20_times_10_pow_1500_l3525_352517


namespace NUMINAMATH_CALUDE_complex_real_part_theorem_l3525_352577

theorem complex_real_part_theorem (a : ℝ) : 
  (((a - Complex.I) / (3 + Complex.I)).re = 1/2) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_theorem_l3525_352577


namespace NUMINAMATH_CALUDE_goods_train_speed_l3525_352511

/-- The speed of the goods train given the conditions of the problem -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 20) 
  (h2 : passing_time = 9) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 92 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_speed_l3525_352511


namespace NUMINAMATH_CALUDE_missing_element_is_loop_l3525_352591

-- Define the basic elements of a flowchart
inductive FlowchartElement
| Input
| Output
| Condition
| Loop

-- Define the program structures
inductive ProgramStructure
| Sequence
| Condition
| Loop

-- Define the known basic elements
def known_elements : List FlowchartElement := [FlowchartElement.Input, FlowchartElement.Output, FlowchartElement.Condition]

-- Define the program structures
def program_structures : List ProgramStructure := [ProgramStructure.Sequence, ProgramStructure.Condition, ProgramStructure.Loop]

-- Theorem: The missing basic element of a flowchart is Loop
theorem missing_element_is_loop : 
  ∃ (e : FlowchartElement), e ∉ known_elements ∧ e = FlowchartElement.Loop :=
sorry

end NUMINAMATH_CALUDE_missing_element_is_loop_l3525_352591


namespace NUMINAMATH_CALUDE_noa_score_l3525_352566

/-- Proves that Noa scored 30 points given the conditions of the problem -/
theorem noa_score (noa_score : ℕ) (phillip_score : ℕ) : 
  phillip_score = 2 * noa_score →
  noa_score + phillip_score = 90 →
  noa_score = 30 := by
sorry

end NUMINAMATH_CALUDE_noa_score_l3525_352566


namespace NUMINAMATH_CALUDE_twentieth_term_is_59_l3525_352571

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 20th term of the arithmetic sequence with first term 2 and common difference 3 is 59 -/
theorem twentieth_term_is_59 :
  arithmeticSequenceTerm 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_59_l3525_352571


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sixth_root_64_l3525_352510

theorem cube_root_27_times_fourth_root_81_times_sixth_root_64 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^6 = 64 ∧ a * b * c = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sixth_root_64_l3525_352510


namespace NUMINAMATH_CALUDE_shooting_game_probability_l3525_352581

-- Define the probability of hitting the target
variable (p : ℝ)

-- Define the number of shooting attempts
def η : ℕ → ℝ
| 1 => p
| 2 => (1 - p) * p
| 3 => (1 - p)^2
| _ => 0

-- Define the expected value of η
def E_η : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p)^2

-- Theorem statement
theorem shooting_game_probability (h1 : 0 < p) (h2 : p < 1) (h3 : E_η > 7/4) :
  p ∈ Set.Ioo 0 (1/2) :=
sorry

end NUMINAMATH_CALUDE_shooting_game_probability_l3525_352581


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l3525_352506

theorem least_addition_for_divisibility (n : Nat) (d : Nat) : ∃ (x : Nat), x < d ∧ (n + x) % d = 0 :=
by
  -- The proof would go here
  sorry

theorem least_addition_to_3198_for_divisibility_by_8 :
  ∃ (x : Nat), x < 8 ∧ (3198 + x) % 8 = 0 ∧ ∀ (y : Nat), y < x → (3198 + y) % 8 ≠ 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_3198_for_divisibility_by_8_l3525_352506


namespace NUMINAMATH_CALUDE_probability_less_than_10_l3525_352552

theorem probability_less_than_10 (p_10_ring : ℝ) (h1 : p_10_ring = 0.22) :
  1 - p_10_ring = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_10_l3525_352552


namespace NUMINAMATH_CALUDE_inequality_and_bound_l3525_352519

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem inequality_and_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = Real.sqrt 2) :
  (∀ x, f x > 3 - |x + 2| ↔ x < -3 ∨ x > 0) ∧
  (∀ x, f x - |x| ≤ Real.sqrt (a^2 + 4*b^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_bound_l3525_352519


namespace NUMINAMATH_CALUDE_qin_jiushao_count_for_specific_polynomial_l3525_352582

/-- The "Qin Jiushao" algorithm for polynomial evaluation -/
def qin_jiushao_eval (coeffs : List ℝ) (x : ℝ) : ℝ := sorry

/-- Counts the number of multiplications and additions in the "Qin Jiushao" algorithm -/
def qin_jiushao_count (coeffs : List ℝ) : (ℕ × ℕ) := sorry

theorem qin_jiushao_count_for_specific_polynomial :
  let coeffs := [5, 4, 3, 2, 1, 1]
  qin_jiushao_count coeffs = (5, 5) := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_count_for_specific_polynomial_l3525_352582


namespace NUMINAMATH_CALUDE_finite_solutions_equation_l3525_352513

theorem finite_solutions_equation :
  ∃ (S : Finset (ℕ × ℕ)), ∀ m n : ℕ,
    m^2 + 2 * 3^n = m * (2^(n+1) - 1) → (m, n) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_equation_l3525_352513


namespace NUMINAMATH_CALUDE_conference_handshakes_l3525_352588

theorem conference_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3525_352588


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l3525_352560

/-- Represents the number of socks of each color in the drawer -/
def socksPerColor : ℕ := 24

/-- Represents the total number of colors of socks in the drawer -/
def numColors : ℕ := 2

/-- Represents the minimum number of socks that must be picked to guarantee a pair of the same color -/
def minSocksToPick : ℕ := 3

/-- Theorem stating that picking 3 socks guarantees at least one pair of the same color,
    and this is the minimum number required -/
theorem min_socks_for_pair :
  (∀ (picked : Finset ℕ), picked.card = minSocksToPick → 
    ∃ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card ≥ 2) ∧
  (∀ (n : ℕ), n < minSocksToPick → 
    ∃ (picked : Finset ℕ), picked.card = n ∧ 
      ∀ (color : Fin numColors), (picked.filter (λ sock => sock % numColors = color)).card < 2) :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l3525_352560


namespace NUMINAMATH_CALUDE_three_squares_sum_l3525_352553

theorem three_squares_sum (n : ℤ) : 3*(n-1)^2 + 8 = (n-3)^2 + (n-1)^2 + (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_three_squares_sum_l3525_352553


namespace NUMINAMATH_CALUDE_spinach_not_music_lover_l3525_352545

-- Define the universe
variable (U : Type)

-- Define predicates
variable (S : U → Prop)  -- x likes spinach
variable (G : U → Prop)  -- x is a pearl diver
variable (Z : U → Prop)  -- x is a music lover

-- State the theorem
theorem spinach_not_music_lover 
  (h1 : ∃ x, S x ∧ ¬G x)
  (h2 : ∀ x, Z x → (G x ∨ ¬S x))
  (h3 : (∀ x, ¬G x → Z x) ∨ (∀ x, G x → ¬Z x))
  : ∀ x, S x → ¬Z x :=
by sorry

end NUMINAMATH_CALUDE_spinach_not_music_lover_l3525_352545


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l3525_352575

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π) : 
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α^2 - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l3525_352575


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3525_352583

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    ∃ (f₁ f₂ : ℝ × ℝ), 
      f₁.1 = focal_length ∧ f₁.2 = 0 ∧
      f₂.1 = -focal_length ∧ f₂.2 = 0 ∧
      ∀ p : ℝ × ℝ, p.1^2/a^2 - p.2^2/b^2 = 1 → 
        Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
        Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 2*a :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_focal_length_l3525_352583


namespace NUMINAMATH_CALUDE_Q_bounds_l3525_352544

/-- The equation of the given curve -/
def curve_equation (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

/-- The expression we want to bound -/
def Q (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

/-- Theorem stating the bounds of Q for points on the curve -/
theorem Q_bounds :
  ∀ x y : ℝ, curve_equation x y → 3 ≤ Q x y ∧ Q x y ≤ 124 :=
by sorry

end NUMINAMATH_CALUDE_Q_bounds_l3525_352544


namespace NUMINAMATH_CALUDE_sin_135_degrees_l3525_352532

theorem sin_135_degrees :
  Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l3525_352532


namespace NUMINAMATH_CALUDE_subtracted_amount_l3525_352505

theorem subtracted_amount (N : ℝ) (A : ℝ) (h1 : N = 100) (h2 : 0.8 * N - A = 60) : A = 20 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l3525_352505


namespace NUMINAMATH_CALUDE_total_pencils_l3525_352559

/-- Given that each child has 2 pencils and there are 15 children, 
    prove that the total number of pencils is 30. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 15) : 
  pencils_per_child * num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3525_352559


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3525_352573

theorem fraction_sum_theorem (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4) 
  (h2 : a/x + b/y + c/z = 3) 
  (h3 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 + 6*(x*y*z)/(a*b*c) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3525_352573


namespace NUMINAMATH_CALUDE_stratified_sampling_and_probability_l3525_352554

-- Define the total number of students
def total_students : ℕ := 350

-- Define the number of students excellent in Chinese
def excellent_chinese : ℕ := 200

-- Define the number of students excellent in English
def excellent_english : ℕ := 150

-- Define the probability of being excellent in both subjects
def prob_both_excellent : ℚ := 1 / 6

-- Define the number of students selected for the sample
def sample_size : ℕ := 6

-- Define the function to calculate the number of students in each category
def calculate_category_sizes : ℕ × ℕ × ℕ := sorry

-- Define the function to calculate the probability of selecting two students with excellent Chinese scores
def calculate_probability : ℚ := sorry

-- Theorem statement
theorem stratified_sampling_and_probability :
  let (a, b, c) := calculate_category_sizes
  (a = 3 ∧ b = 2 ∧ c = 1) ∧ calculate_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_and_probability_l3525_352554


namespace NUMINAMATH_CALUDE_max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l3525_352541

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0  -- Define a default value for x outside the given ranges

-- Theorem statements
theorem max_receptivity (x : ℝ) :
  (∀ x, f x ≤ 60) ∧
  (f 10 = 60) ∧
  (∀ x, 10 < x → x ≤ 15 → f x = 60) :=
sorry

theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

theorem no_continuous_high_receptivity :
  ¬ ∃ a b : ℝ, b - a = 12 ∧ ∀ x, a ≤ x ∧ x ≤ b → f x ≥ 56 :=
sorry

end NUMINAMATH_CALUDE_max_receptivity_receptivity_comparison_no_continuous_high_receptivity_l3525_352541


namespace NUMINAMATH_CALUDE_abs_z_equals_five_l3525_352580

theorem abs_z_equals_five (z : ℂ) (h : z - 3 = (3 + I) / I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_five_l3525_352580


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3525_352526

/-- An ellipse with given major axis and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
inductive StandardEquation where
  | x_axis : StandardEquation
  | y_axis : StandardEquation

/-- Theorem: For an ellipse with major axis 8 and eccentricity 3/4, 
    its standard equation is either (x²/16) + (y²/7) = 1 or (x²/7) + (y²/16) = 1 -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.major_axis = 8) 
  (h2 : e.eccentricity = 3/4) :
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.x_axis → ∀ (x y : ℝ), x^2/16 + y^2/7 = 1) ∧ 
    (eq = StandardEquation.y_axis → ∀ (x y : ℝ), x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3525_352526


namespace NUMINAMATH_CALUDE_equation_is_ellipse_l3525_352518

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0

-- Define what it means for the equation to represent an ellipse
def is_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ), 
    A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), eq x y ↔ ((x - h)^2 / A + (y - k)^2 / B = 1)

-- Theorem statement
theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_ellipse_l3525_352518


namespace NUMINAMATH_CALUDE_simplify_expression_l3525_352579

theorem simplify_expression (x y : ℝ) : (2 * x^2 - x * y) - (x^2 + x * y - 8) = x^2 - 2 * x * y + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3525_352579


namespace NUMINAMATH_CALUDE_sum_of_roots_x4_minus_4x3_minus_1_l3525_352522

theorem sum_of_roots_x4_minus_4x3_minus_1 : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, x^4 - 4*x^3 - 1 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    r₁ + r₂ + r₃ + r₄ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_x4_minus_4x3_minus_1_l3525_352522


namespace NUMINAMATH_CALUDE_constant_value_l3525_352593

theorem constant_value (x y z : ℝ) : 
  ∃ (c : ℝ), ∀ (x y z : ℝ), 
    ((x - y)^3 + (y - z)^3 + (z - x)^3) / (c * (x - y) * (y - z) * (z - x)) = 0.2 → c = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l3525_352593


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3525_352512

theorem max_area_rectangle (perimeter : ℕ) (area : ℕ → ℕ → ℕ) :
  perimeter = 150 →
  (∀ w h : ℕ, area w h = w * h) →
  (∀ w h : ℕ, 2 * w + 2 * h = perimeter → area w h ≤ 1406) ∧
  (∃ w h : ℕ, 2 * w + 2 * h = perimeter ∧ area w h = 1406) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3525_352512


namespace NUMINAMATH_CALUDE_target_hit_probability_l3525_352562

theorem target_hit_probability (p_a p_b p_c : ℚ) 
  (h_a : p_a = 1/2) 
  (h_b : p_b = 1/3) 
  (h_c : p_c = 1/4) : 
  1 - (1 - p_a) * (1 - p_b) * (1 - p_c) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3525_352562


namespace NUMINAMATH_CALUDE_initial_average_production_l3525_352540

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) 
  (h1 : n = 19)
  (h2 : today_production = 90)
  (h3 : new_average = 52) : 
  ∃ A : ℚ, A = 50 ∧ (A * n + today_production) / (n + 1) = new_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_production_l3525_352540


namespace NUMINAMATH_CALUDE_revenue_change_l3525_352529

theorem revenue_change 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : price_increase = 0.75) 
  (h2 : quantity_decrease = 0.45) : 
  let new_price := original_price * (1 + price_increase)
  let new_quantity := original_quantity * (1 - quantity_decrease)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = -0.0375 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l3525_352529


namespace NUMINAMATH_CALUDE_greatest_common_factor_48_180_240_l3525_352515

theorem greatest_common_factor_48_180_240 : Nat.gcd 48 (Nat.gcd 180 240) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_48_180_240_l3525_352515


namespace NUMINAMATH_CALUDE_find_n_l3525_352599

theorem find_n : ∃ n : ℤ, 3^3 - 5 = 2^5 + n ∧ n = -10 := by sorry

end NUMINAMATH_CALUDE_find_n_l3525_352599


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l3525_352550

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_seq : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l3525_352550


namespace NUMINAMATH_CALUDE_milk_production_l3525_352504

/-- Given that x cows produce y gallons of milk in z days, 
    calculate the amount of milk w cows produce in v days with 10% daily waste. -/
theorem milk_production (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0) :
  let daily_waste : ℝ := 0.1
  let milk_per_cow_per_day : ℝ := y / (z * x)
  let effective_milk_per_cow_per_day : ℝ := milk_per_cow_per_day * (1 - daily_waste)
  effective_milk_per_cow_per_day * w * v = 0.9 * (w * y * v) / (z * x) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l3525_352504


namespace NUMINAMATH_CALUDE_sphere_surface_area_containing_unit_cube_l3525_352537

/-- The surface area of a sphere that contains all eight vertices of a unit cube -/
theorem sphere_surface_area_containing_unit_cube : ℝ := by
  -- Define a cube with edge length 1
  let cube_edge_length : ℝ := 1

  -- Define the sphere that contains all vertices of the cube
  let sphere_radius : ℝ := (Real.sqrt 3) / 2

  -- Define the surface area of the sphere
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

  -- Prove that the surface area equals 3π
  have : sphere_surface_area = 3 * Real.pi := by sorry

  -- Return the result
  exact 3 * Real.pi


end NUMINAMATH_CALUDE_sphere_surface_area_containing_unit_cube_l3525_352537


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l3525_352587

theorem tangent_line_cubic_curve (m : ℝ) : 
  (∃ x y : ℝ, y = 12 * x + m ∧ y = x^3 - 2 ∧ 12 = 3 * x^2) → 
  (m = -18 ∨ m = 14) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l3525_352587


namespace NUMINAMATH_CALUDE_arithmetic_progression_equiv_square_product_l3525_352508

theorem arithmetic_progression_equiv_square_product 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ d : ℝ, Real.log y - Real.log x = d ∧ Real.log z - Real.log y = d) ↔ 
  y^2 = x*z := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equiv_square_product_l3525_352508


namespace NUMINAMATH_CALUDE_circle_radius_when_perimeter_equals_area_l3525_352524

/-- Given a square and its circumscribed circle, if the perimeter of the square in inches
    equals the area of the circle in square inches, then the radius of the circle is 8/π inches. -/
theorem circle_radius_when_perimeter_equals_area (s : ℝ) (r : ℝ) :
  s > 0 → r > 0 → s = 2 * r → 4 * s = π * r^2 → r = 8 / π :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_when_perimeter_equals_area_l3525_352524


namespace NUMINAMATH_CALUDE_same_type_book_probability_l3525_352546

/-- The probability of selecting two books of the same type from a collection of novels and biographies -/
theorem same_type_book_probability (novels biographies : ℕ) 
  (h_novels : novels = 12) (h_biographies : biographies = 9) : 
  (Nat.choose novels 2 + Nat.choose biographies 2) / Nat.choose (novels + biographies) 2 = 102 / 210 := by
  sorry

end NUMINAMATH_CALUDE_same_type_book_probability_l3525_352546


namespace NUMINAMATH_CALUDE_total_haircut_time_l3525_352520

/-- The time it takes to cut a woman's hair in minutes -/
def womanHairCutTime : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def manHairCutTime : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kidHairCutTime : ℕ := 25

/-- The number of women's haircuts Joe performed -/
def numWomenHaircuts : ℕ := 3

/-- The number of men's haircuts Joe performed -/
def numMenHaircuts : ℕ := 2

/-- The number of kids' haircuts Joe performed -/
def numKidsHaircuts : ℕ := 3

/-- Theorem stating the total time Joe spent cutting hair -/
theorem total_haircut_time :
  numWomenHaircuts * womanHairCutTime +
  numMenHaircuts * manHairCutTime +
  numKidsHaircuts * kidHairCutTime = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_haircut_time_l3525_352520


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l3525_352536

open Real
open EuclideanSpace

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the vectors
variable (MA MB MC : V)

-- State the theorem
theorem vectors_are_coplanar 
  (h_noncollinear : ¬ ∃ (k : ℝ), MA = k • MB)
  (h_MC_def : MC = 5 • MA - 3 • MB) :
  ∃ (a b c : ℝ), a • MA + b • MB + c • MC = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l3525_352536


namespace NUMINAMATH_CALUDE_residue_of_power_mod_13_l3525_352500

theorem residue_of_power_mod_13 : (5 ^ 1234 : ℕ) % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_power_mod_13_l3525_352500


namespace NUMINAMATH_CALUDE_sin_neg_seven_pi_sixth_l3525_352564

theorem sin_neg_seven_pi_sixth : Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_seven_pi_sixth_l3525_352564


namespace NUMINAMATH_CALUDE_cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l3525_352503

-- Problem 1
theorem cos_squared_minus_three_sin_cos (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  let P : ℝ × ℝ := (m, 3 * m)
  (Real.cos α)^2 - 3 * (Real.sin α) * (Real.cos α) = -4/5 := by sorry

-- Problem 2
theorem angle_in_second_quadrant (θ : ℝ) (a : ℝ) 
  (h1 : Real.sin θ = (1 - a) / (1 + a))
  (h2 : Real.cos θ = (3 * a - 1) / (1 + a))
  (h3 : 0 < Real.sin θ ∧ Real.cos θ < 0) :
  a = 1/9 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_three_sin_cos_angle_in_second_quadrant_l3525_352503


namespace NUMINAMATH_CALUDE_factor_expression_l3525_352516

theorem factor_expression (x y : ℝ) : 286 * x^2 * y + 143 * x = 143 * x * (2 * x * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3525_352516


namespace NUMINAMATH_CALUDE_mod_equiv_unique_solution_l3525_352555

theorem mod_equiv_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2357 ≡ n [ZMOD 9] :=
by sorry

end NUMINAMATH_CALUDE_mod_equiv_unique_solution_l3525_352555


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3525_352595

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 8) :
  α^5 + β^5 + γ^5 = 46.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3525_352595


namespace NUMINAMATH_CALUDE_concert_revenue_l3525_352557

theorem concert_revenue (ticket_price : ℝ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (first_discount : ℝ) (second_discount : ℝ) (total_buyers : ℕ) :
  ticket_price = 20 →
  first_group_size = 10 →
  second_group_size = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  total_buyers = 56 →
  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_buyers := total_buyers - first_group_size - second_group_size
  let remaining_revenue := remaining_buyers * ticket_price
  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue
  total_revenue = 980 := by sorry

end NUMINAMATH_CALUDE_concert_revenue_l3525_352557


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l3525_352514

theorem min_value_sum_of_roots (x : ℝ) :
  let y := Real.sqrt (x^2 - 2*x + 2) + Real.sqrt (x^2 - 10*x + 34)
  y ≥ 4 * Real.sqrt 2 ∧ ∃ x₀ : ℝ, Real.sqrt (x₀^2 - 2*x₀ + 2) + Real.sqrt (x₀^2 - 10*x₀ + 34) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l3525_352514


namespace NUMINAMATH_CALUDE_sequence_expression_l3525_352525

theorem sequence_expression (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l3525_352525


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l3525_352574

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 4 → 
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l3525_352574


namespace NUMINAMATH_CALUDE_f_of_7_eq_17_l3525_352542

/-- The polynomial function f(x) = 2x^4 - 17x^3 + 26x^2 - 24x - 60 -/
def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

/-- Theorem: The value of f(7) is 17 -/
theorem f_of_7_eq_17 : f 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_of_7_eq_17_l3525_352542


namespace NUMINAMATH_CALUDE_total_gray_trees_count_l3525_352585

/-- Represents an aerial photo with tree counts -/
structure AerialPhoto where
  totalTrees : ℕ
  whiteTrees : ℕ

/-- Calculates the number of trees in the gray area of a photo -/
def grayTrees (photo : AerialPhoto) : ℕ :=
  photo.totalTrees - photo.whiteTrees

theorem total_gray_trees_count 
  (photo1 photo2 photo3 : AerialPhoto)
  (h1 : photo1.totalTrees = 100)
  (h2 : photo1.whiteTrees = 82)
  (h3 : photo2.totalTrees = 90)
  (h4 : photo2.whiteTrees = 82)
  (h5 : photo3.whiteTrees = 75)
  (h6 : photo1.totalTrees = photo2.totalTrees)
  (h7 : photo2.totalTrees = photo3.totalTrees) :
  grayTrees photo1 + grayTrees photo2 + grayTrees photo3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_gray_trees_count_l3525_352585


namespace NUMINAMATH_CALUDE_max_product_with_constraints_l3525_352548

theorem max_product_with_constraints (a b : ℕ) :
  a + b = 100 →
  a % 5 = 2 →
  b % 6 = 3 →
  a * b ≤ 2331 ∧ ∃ (a' b' : ℕ), a' + b' = 100 ∧ a' % 5 = 2 ∧ b' % 6 = 3 ∧ a' * b' = 2331 :=
by sorry

end NUMINAMATH_CALUDE_max_product_with_constraints_l3525_352548


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l3525_352528

/-- 
Theorem: The smallest value of n for which 5x^2 + nx + 60 can be factored 
as the product of two linear factors with integer coefficients is 56.
-/
theorem smallest_n_for_factorization : 
  (∃ n : ℤ, ∀ m : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d))) ∧
  (∀ n : ℤ, 
    (∃ a b : ℤ, 5 * X^2 + n * X + 60 = (5 * X + a) * (X + b)) ∧ 
    (∀ k : ℤ, k < n → ¬∃ c d : ℤ, 5 * X^2 + k * X + 60 = (5 * X + c) * (X + d)) 
    → n = 56) :=
sorry


end NUMINAMATH_CALUDE_smallest_n_for_factorization_l3525_352528


namespace NUMINAMATH_CALUDE_unique_right_triangle_exists_l3525_352558

theorem unique_right_triangle_exists : ∃! (a : ℝ), 
  a > 0 ∧ 
  let b := 2 * a
  let c := Real.sqrt (a^2 + b^2)
  (a + b + c) - (1/2 * a * b) = c :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_exists_l3525_352558


namespace NUMINAMATH_CALUDE_all_multiples_contain_two_l3525_352576

def numbers : List ℕ := [418, 244, 816, 426, 24]

def containsTwo (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 2

theorem all_multiples_contain_two :
  ∀ n ∈ numbers, containsTwo (3 * n) :=
by sorry

end NUMINAMATH_CALUDE_all_multiples_contain_two_l3525_352576


namespace NUMINAMATH_CALUDE_christinas_walking_speed_l3525_352502

/-- Prove that Christina's walking speed is 8 feet per second given the initial conditions and the total distance traveled by Lindy. -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jack_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : lindy_speed = 10)
  (h4 : lindy_total_distance = 100) :
  ∃ christina_speed : ℝ, christina_speed = 8 ∧ 
    (lindy_total_distance / lindy_speed) * (jack_speed + christina_speed) = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_christinas_walking_speed_l3525_352502


namespace NUMINAMATH_CALUDE_f_16_values_l3525_352527

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * f (a^2 + b^2) = 2 * (f a)^2 + 2 * (f b)^2 - f a * f b

theorem f_16_values (f : ℕ → ℕ) (h : is_valid_f f) : 
  {n : ℕ | f 16 = n} = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_f_16_values_l3525_352527


namespace NUMINAMATH_CALUDE_odd_function_sum_l3525_352535

-- Define an odd function f on the real numbers
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : isOddFunction f) (h2 : f 1 = -2) :
  f (-1) + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3525_352535


namespace NUMINAMATH_CALUDE_solve_for_a_l3525_352567

theorem solve_for_a : ∃ a : ℝ, (2 * (-1) + 3 * a = 4) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l3525_352567


namespace NUMINAMATH_CALUDE_pizza_slices_left_l3525_352523

theorem pizza_slices_left (total_slices : ℕ) (john_slices : ℕ) (sam_multiplier : ℕ) : 
  total_slices = 12 →
  john_slices = 3 →
  sam_multiplier = 2 →
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l3525_352523


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3525_352597

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 780)
  (h2 : rate = 4.166666666666667 / 100)
  (h3 : time = 4) :
  principal * rate * time = 130 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3525_352597


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3525_352572

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = 7) : a^2 + 1/a^2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3525_352572


namespace NUMINAMATH_CALUDE_total_gum_pieces_l3525_352578

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- Theorem: The total number of gum pieces Robin has is 135 -/
theorem total_gum_pieces : num_packages * pieces_per_package = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l3525_352578


namespace NUMINAMATH_CALUDE_trisha_total_distance_l3525_352570

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha's total walking distance is 0.89 miles -/
theorem trisha_total_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by sorry

end NUMINAMATH_CALUDE_trisha_total_distance_l3525_352570


namespace NUMINAMATH_CALUDE_total_amount_spent_l3525_352556

theorem total_amount_spent (num_pens num_pencils : ℕ) 
                           (avg_pen_price avg_pencil_price : ℚ) : 
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 14 →
  avg_pencil_price = 2 →
  (num_pens : ℚ) * avg_pen_price + (num_pencils : ℚ) * avg_pencil_price = 570 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l3525_352556


namespace NUMINAMATH_CALUDE_sufficient_condition_for_ellipse_l3525_352549

/-- The equation of a potential ellipse -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / m + y^2 / (2*m - 1) = 1

/-- Condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- Theorem stating that m > 1 is a sufficient but not necessary condition for the equation to represent an ellipse -/
theorem sufficient_condition_for_ellipse :
  ∀ m : ℝ, m > 1 → is_ellipse m ∧ ∃ m₀ : ℝ, m₀ ≤ 1 ∧ is_ellipse m₀ :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_ellipse_l3525_352549


namespace NUMINAMATH_CALUDE_age_ratio_in_3_years_l3525_352530

def franks_current_age : ℕ := 12
def johns_current_age : ℕ := franks_current_age + 15

def franks_age_in_3_years : ℕ := franks_current_age + 3
def johns_age_in_3_years : ℕ := johns_current_age + 3

theorem age_ratio_in_3_years :
  ∃ (k : ℕ), k > 0 ∧ johns_age_in_3_years = k * franks_age_in_3_years ∧
  johns_age_in_3_years / franks_age_in_3_years = 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_3_years_l3525_352530


namespace NUMINAMATH_CALUDE_curve_transformation_l3525_352533

/-- Given a curve C: (x-y)^2 + y^2 = 1 transformed by matrix A = [[2, -2], [0, 1]],
    prove that the resulting curve C' has the equation x^2/4 + y^2 = 1 -/
theorem curve_transformation (x₀ y₀ x y : ℝ) : 
  (x₀ - y₀)^2 + y₀^2 = 1 →
  x = 2*x₀ - 2*y₀ →
  y = y₀ →
  x^2/4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_curve_transformation_l3525_352533


namespace NUMINAMATH_CALUDE_increasing_function_bounds_l3525_352589

theorem increasing_function_bounds (k : ℕ+) (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ m n : ℕ+, m < n → f m < f n)
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_bounds_l3525_352589


namespace NUMINAMATH_CALUDE_f_min_max_on_I_l3525_352568

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- Define the interval
def I : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem f_min_max_on_I :
  ∃ (min max : ℝ), 
    (∀ x ∈ I, f x ≥ min) ∧ 
    (∃ x ∈ I, f x = min) ∧
    (∀ x ∈ I, f x ≤ max) ∧ 
    (∃ x ∈ I, f x = max) ∧
    min = -64 ∧ max = 0 :=
sorry

end NUMINAMATH_CALUDE_f_min_max_on_I_l3525_352568


namespace NUMINAMATH_CALUDE_barn_paint_area_l3525_352596

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a barn with a dividing wall -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let externalWallArea := 2 * (d.width * d.height + d.length * d.height)
  let dividingWallArea := 2 * (d.width * d.height)
  let ceilingArea := d.width * d.length
  2 * externalWallArea + dividingWallArea + ceilingArea

/-- The dimensions of the barn in the problem -/
def problemBarn : BarnDimensions :=
  { width := 12
  , length := 15
  , height := 5 }

theorem barn_paint_area :
  totalPaintArea problemBarn = 840 := by
  sorry


end NUMINAMATH_CALUDE_barn_paint_area_l3525_352596


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3525_352584

/-- For a cone with an equilateral triangle as its axial section, 
    the angle of the sector formed by unfolding its lateral surface is π radians. -/
theorem cone_lateral_surface_angle (R r : ℝ) (α : ℝ) : 
  R > 0 ∧ r > 0 ∧ R = 2 * r → α = π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3525_352584


namespace NUMINAMATH_CALUDE_square_ratio_sum_l3525_352586

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l3525_352586


namespace NUMINAMATH_CALUDE_base_equation_solution_l3525_352538

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ := sorry

/-- Converts a base-a number to base-10 representation --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ := sorry

/-- Adds two numbers in base-a --/
def addBaseA (n1 : List ℕ) (n2 : List ℕ) (a : ℕ) : List ℕ := sorry

theorem base_equation_solution :
  ∃! a : ℕ, 
    a > 11 ∧ 
    addBaseA (toBaseA 396 a) (toBaseA 574 a) a = toBaseA (96 * 11) a := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l3525_352538


namespace NUMINAMATH_CALUDE_min_tile_A_1011_l3525_352551

/-- Represents a tile type -/
inductive Tile
| A  -- Covers 3 squares: 2 in one row and 1 in the adjacent row
| B  -- Covers 4 squares: 2 in one row and 2 in the adjacent row

/-- Represents a tiling of a square grid -/
def Tiling (n : ℕ) := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Checks if a tiling is valid for an n×n square -/
def isValidTiling (n : ℕ) (t : Tiling n) : Prop := sorry

/-- Counts the number of tiles of type A in a tiling -/
def countTileA (t : Tiling n) : ℕ := sorry

/-- Theorem: The minimum number of tiles A required to tile a 1011×1011 square is 2023 -/
theorem min_tile_A_1011 :
  ∀ t : Tiling 1011, isValidTiling 1011 t → countTileA t ≥ 2023 ∧
  ∃ t' : Tiling 1011, isValidTiling 1011 t' ∧ countTileA t' = 2023 := by
  sorry

#check min_tile_A_1011

end NUMINAMATH_CALUDE_min_tile_A_1011_l3525_352551


namespace NUMINAMATH_CALUDE_not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l3525_352507

theorem not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true
  (h1 : ¬p)
  (h2 : ¬(p ∧ q)) :
  ¬∀ (p q : Prop), p ∨ q :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_p_and_q_implies_p_or_q_not_necessarily_true_l3525_352507


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3525_352539

theorem smallest_solution_of_equation :
  let x : ℝ := (5 - Real.sqrt 33) / 2
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 1) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3525_352539


namespace NUMINAMATH_CALUDE_expand_complex_product_l3525_352590

theorem expand_complex_product (x : ℂ) : (x + Complex.I) * (x - 7) = x^2 - 7*x + Complex.I*x - 7*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_expand_complex_product_l3525_352590


namespace NUMINAMATH_CALUDE_smallest_n_for_multiples_l3525_352501

theorem smallest_n_for_multiples : ∃ (a : Fin 15 → ℕ), 
  (∀ i : Fin 15, 16 ≤ a i ∧ a i ≤ 34) ∧ 
  (∀ i : Fin 15, a i % (i.val + 1) = 0) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j) ∧
  (∀ n : ℕ, n < 34 → ¬∃ (b : Fin 15 → ℕ), 
    (∀ i : Fin 15, 16 ≤ b i ∧ b i ≤ n) ∧ 
    (∀ i : Fin 15, b i % (i.val + 1) = 0) ∧
    (∀ i j : Fin 15, i ≠ j → b i ≠ b j)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_multiples_l3525_352501


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3525_352531

theorem negation_of_proposition (P : ℝ → Prop) : 
  (∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ¬(∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3525_352531


namespace NUMINAMATH_CALUDE_correct_elderly_sample_size_l3525_352543

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the number of elderly people to be sampled -/
def elderlySampleSize (p : Population) (sampleSize : ℕ) : ℕ :=
  (p.elderly * sampleSize) / totalPopulation p

theorem correct_elderly_sample_size (p : Population) (sampleSize : ℕ) 
  (h1 : p.elderly = 30)
  (h2 : p.middleAged = 90)
  (h3 : p.young = 60)
  (h4 : sampleSize = 36) :
  elderlySampleSize p sampleSize = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_elderly_sample_size_l3525_352543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3525_352534

/-- Given two arithmetic sequences and their sum ratios, prove a specific ratio of their terms -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h : ∀ n : ℕ, S n / T n = (2 * n - 3 : ℚ) / (4 * n - 1 : ℚ)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3525_352534


namespace NUMINAMATH_CALUDE_working_partner_receives_8160_l3525_352569

/-- Calculates the money received by the working partner in a business partnership --/
def money_received_by_working_partner (a_investment : ℕ) (b_investment : ℕ) (management_fee_percent : ℕ) (total_profit : ℕ) : ℕ :=
  let management_fee := (management_fee_percent * total_profit) / 100
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_share := (a_investment * remaining_profit) / total_investment
  management_fee + a_share

/-- Theorem stating that under given conditions, the working partner receives 8160 rs --/
theorem working_partner_receives_8160 :
  money_received_by_working_partner 5000 1000 10 9600 = 8160 := by
  sorry

#eval money_received_by_working_partner 5000 1000 10 9600

end NUMINAMATH_CALUDE_working_partner_receives_8160_l3525_352569


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l3525_352521

-- Part 1
theorem inequality_one (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a*b + b*c + c*a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l3525_352521


namespace NUMINAMATH_CALUDE_polynomial_roots_degree_zero_l3525_352565

theorem polynomial_roots_degree_zero (F : Type*) [Field F] :
  ∀ (P : Polynomial F),
    (∃ (S : Finset F), (∀ x ∈ S, P.eval x = 0) ∧ S.card > P.degree) →
    P = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_degree_zero_l3525_352565


namespace NUMINAMATH_CALUDE_sum_of_squares_for_specific_conditions_l3525_352592

theorem sum_of_squares_for_specific_conditions : 
  ∃ (S : Finset ℕ), 
    (∀ s ∈ S, ∃ x y z : ℕ, 
      x > 0 ∧ y > 0 ∧ z > 0 ∧
      x + y + z = 30 ∧ 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 ∧
      s = x^2 + y^2 + z^2) ∧
    (∀ x y z : ℕ, 
      x > 0 → y > 0 → z > 0 →
      x + y + z = 30 → 
      Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12 →
      (x^2 + y^2 + z^2) ∈ S) ∧
    S.sum id = 710 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_for_specific_conditions_l3525_352592


namespace NUMINAMATH_CALUDE_students_in_both_chorus_and_band_l3525_352563

theorem students_in_both_chorus_and_band :
  ∀ (total chorus band neither both : ℕ),
    total = 50 →
    chorus = 18 →
    band = 26 →
    neither = 8 →
    total = chorus + band - both + neither →
    both = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_both_chorus_and_band_l3525_352563


namespace NUMINAMATH_CALUDE_equal_chord_lengths_l3525_352598

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the condition D^2 ≠ E^2 > 4F
def condition (D E F : ℝ) : Prop :=
  D^2 ≠ E^2 ∧ E^2 > 4*F

-- Theorem statement
theorem equal_chord_lengths (D E F : ℝ) 
  (h : condition D E F) : 
  ∃ (chord_x chord_y : ℝ), 
    (∀ (x y : ℝ), circle_equation x y D E F → 
      (x = chord_x/2 ∨ x = -chord_x/2) ∨ (y = chord_y/2 ∨ y = -chord_y/2)) ∧
    chord_x = chord_y :=
sorry

end NUMINAMATH_CALUDE_equal_chord_lengths_l3525_352598


namespace NUMINAMATH_CALUDE_matrix_power_four_l3525_352561

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -2; 2, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-7 : ℤ), 24; -24, 7] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3525_352561
