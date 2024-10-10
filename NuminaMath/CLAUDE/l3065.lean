import Mathlib

namespace fraction_integer_iff_q_values_l3065_306568

theorem fraction_integer_iff_q_values (q : ℕ+) :
  (∃ (k : ℕ+), (4 * q + 28 : ℚ) / (3 * q - 7 : ℚ) = k) ↔ q ∈ ({7, 15, 25} : Set ℕ+) := by
  sorry

end fraction_integer_iff_q_values_l3065_306568


namespace equality_of_coefficients_l3065_306556

theorem equality_of_coefficients (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a ∧ 
                b * x^2 + c * x + a ≥ c * x^2 + a * x + b) :
  a = b ∧ b = c := by
  sorry

end equality_of_coefficients_l3065_306556


namespace simplest_form_sqrt_l3065_306521

/-- A number is a perfect square if it's the product of an integer with itself -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A square root is in simplest form if it cannot be simplified further -/
def is_simplest_form (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, n = a * b ∧ is_perfect_square a ∧ b > 1)

/-- The square root of a fraction is in simplest form if it cannot be simplified further -/
def is_simplest_form_frac (n d : ℕ) : Prop :=
  ¬(∃ a b c : ℕ, n = a * b ∧ d = a * c ∧ is_perfect_square a ∧ (b > 1 ∨ c > 1))

theorem simplest_form_sqrt :
  is_simplest_form 14 ∧
  ¬is_simplest_form 12 ∧
  ¬is_simplest_form 8 ∧
  ¬is_simplest_form_frac 1 3 :=
sorry

end simplest_form_sqrt_l3065_306521


namespace no_solutions_for_equation_l3065_306548

theorem no_solutions_for_equation :
  ∀ (x y : ℝ), x^2 + y^2 - 2*y + 2 ≠ 0 := by
  sorry

end no_solutions_for_equation_l3065_306548


namespace tangent_half_angle_identity_l3065_306578

theorem tangent_half_angle_identity (α : Real) (m : Real) 
  (h : Real.tan (α / 2) = m) : 
  (1 - 2 * Real.sin (α / 2) ^ 2) / (1 + Real.sin α) = (1 - m) / (1 + m) := by
  sorry

end tangent_half_angle_identity_l3065_306578


namespace exponent_of_p_in_product_l3065_306562

theorem exponent_of_p_in_product (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  ∃ (a b : ℕ), (a + 1) * (b + 1) = 32 ∧ a = 3 := by
  sorry

end exponent_of_p_in_product_l3065_306562


namespace box_volume_formula_l3065_306525

/-- The volume of a box formed by cutting squares from corners of a metal sheet -/
def boxVolume (x : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*x) * x

theorem box_volume_formula (x : ℝ) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end box_volume_formula_l3065_306525


namespace eighteen_wheel_truck_toll_l3065_306569

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.5 + 0.5 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (total_wheels : ℕ) : ℕ :=
  1 + (total_wheels - 2) / 4

theorem eighteen_wheel_truck_toll :
  toll (axles_count 18) = 3 := by
  sorry

end eighteen_wheel_truck_toll_l3065_306569


namespace product_of_two_positive_quantities_l3065_306554

theorem product_of_two_positive_quantities (s : ℝ) (h : s > 0) :
  ¬(∀ x : ℝ, 0 < x → x < s → 
    (x * (s - x) ≤ y * (s - y) → (x = 0 ∨ x = s))) :=
sorry

end product_of_two_positive_quantities_l3065_306554


namespace curve_self_intersection_l3065_306508

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 3, t^3 - 6*t + 2)

-- Theorem statement
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ curve t₁ = p ∧ curve t₂ = p ∧ p = (3, 2) := by
  sorry

end curve_self_intersection_l3065_306508


namespace square_area_difference_l3065_306596

/-- Given two squares ABCD and EGFO with the specified conditions, 
    prove that the difference between their areas is 11.5 -/
theorem square_area_difference (a b : ℕ+) 
  (h1 : (a.val : ℝ)^2 / 2 - (b.val : ℝ)^2 / 2 = 3.25) 
  (h2 : (b.val : ℝ) > (a.val : ℝ)) : 
  (a.val : ℝ)^2 - (b.val : ℝ)^2 = -11.5 := by
  sorry

end square_area_difference_l3065_306596


namespace hexagon_side_length_l3065_306559

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 48) : 
  perimeter / 6 = 8 := by
  sorry

end hexagon_side_length_l3065_306559


namespace consecutive_odd_numbers_average_l3065_306565

theorem consecutive_odd_numbers_average (a b c d : ℕ) : 
  a = 27 ∧ 
  b = a - 2 ∧ 
  c = b - 2 ∧ 
  d = c - 2 ∧ 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d → 
  (a + b + c + d) / 4 = 24 := by
sorry

end consecutive_odd_numbers_average_l3065_306565


namespace parallelogram_area_l3065_306523

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  let base : ℝ := 20
  let height : ℝ := 4
  let area := base * height
  area = 80 := by sorry

end parallelogram_area_l3065_306523


namespace distribute_six_balls_three_boxes_l3065_306506

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end distribute_six_balls_three_boxes_l3065_306506


namespace sum_of_cubes_consecutive_integers_l3065_306503

theorem sum_of_cubes_consecutive_integers :
  ∃ n : ℤ, n^3 + (n + 1)^3 = 9 :=
sorry

end sum_of_cubes_consecutive_integers_l3065_306503


namespace translation_right_4_units_l3065_306529

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the x-direction -/
def translateX (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

theorem translation_right_4_units (P : Point) (P' : Point) :
  P.x = -2 ∧ P.y = 3 →
  P' = translateX P 4 →
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end translation_right_4_units_l3065_306529


namespace two_heads_probability_l3065_306534

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def allOutcomes : Finset TwoCoinsOutcome := sorry

/-- The set of outcomes where both coins show heads -/
def twoHeadsOutcomes : Finset TwoCoinsOutcome := sorry

/-- Proposition: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability :
  (Finset.card twoHeadsOutcomes) / (Finset.card allOutcomes : ℚ) = 1 / 4 := by sorry

end two_heads_probability_l3065_306534


namespace parallel_vectors_l3065_306583

/-- Two vectors in ℝ² -/
def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Theorem: If a(m) is parallel to b, then m = -3/2 -/
theorem parallel_vectors (m : ℝ) :
  parallel (a m) b → m = -3/2 := by sorry

end parallel_vectors_l3065_306583


namespace bus_stop_walk_time_l3065_306598

theorem bus_stop_walk_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h : usual_speed > 0) 
  (h1 : usual_time > 0)
  (h2 : (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time) : 
  usual_time = 30 := by
sorry

end bus_stop_walk_time_l3065_306598


namespace range_of_b_l3065_306572

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem range_of_b (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  (∃ (zs : Finset ℝ), zs.card = 5 ∧ ∀ z ∈ zs, z ∈ Set.Icc (-2) 2 ∧ f z = 0) →
  b ∈ Set.Ioo (1/4) 1 ∪ {5/4} :=
by sorry

end range_of_b_l3065_306572


namespace simplify_expressions_l3065_306501

variable (a b : ℝ)

theorem simplify_expressions :
  (-2 * a * b - a^2 + 3 * a * b - 5 * a^2 = a * b - 6 * a^2) ∧
  ((4 * a * b - b^2) - 2 * (a^2 + 2 * a * b - b^2) = b^2 - 2 * a^2) :=
by sorry

end simplify_expressions_l3065_306501


namespace fixed_point_of_exponential_function_l3065_306512

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 :=
by sorry

end fixed_point_of_exponential_function_l3065_306512


namespace fraction_multiplication_l3065_306543

theorem fraction_multiplication : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 = 2 := by
  sorry

end fraction_multiplication_l3065_306543


namespace max_n_satisfying_condition_l3065_306538

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S (n : ℕ) : ℕ := 2 * sequence_a n - n

theorem max_n_satisfying_condition :
  (∀ n : ℕ, sum_S n = 2 * sequence_a n - n) →
  (∃ max_n : ℕ, (∀ n : ℕ, n ≤ max_n ↔ sequence_a n ≤ 10 * n) ∧ max_n = 5) :=
by sorry

end max_n_satisfying_condition_l3065_306538


namespace andy_work_hours_l3065_306509

-- Define the variables and constants
def hourly_rate : ℝ := 9
def restring_fee : ℝ := 15
def grommet_fee : ℝ := 10
def stencil_fee : ℝ := 1
def total_earnings : ℝ := 202
def racquets_strung : ℕ := 7
def grommets_changed : ℕ := 2
def stencils_painted : ℕ := 5

-- State the theorem
theorem andy_work_hours :
  ∃ (hours : ℝ),
    hours * hourly_rate +
    racquets_strung * restring_fee +
    grommets_changed * grommet_fee +
    stencils_painted * stencil_fee = total_earnings ∧
    hours = 8 := by sorry

end andy_work_hours_l3065_306509


namespace rectangle_area_l3065_306549

theorem rectangle_area (l w : ℝ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) :
  l * w = 150 :=
sorry

end rectangle_area_l3065_306549


namespace simple_interest_rate_l3065_306591

/-- Proves that given a principal of $600 lent at simple interest for 8 years,
    if the total interest is $360 less than the principal,
    then the annual interest rate is 5%. -/
theorem simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) (rate : ℝ) :
  principal = 600 →
  time = 8 →
  interest = principal - 360 →
  interest = principal * rate * time →
  rate = 0.05 := by sorry

end simple_interest_rate_l3065_306591


namespace emily_quiz_score_l3065_306531

theorem emily_quiz_score (scores : List ℝ) (target_mean : ℝ) : 
  scores = [92, 95, 87, 89, 100] →
  target_mean = 93 →
  let new_score := 95
  let all_scores := scores ++ [new_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
sorry


end emily_quiz_score_l3065_306531


namespace function_through_points_l3065_306577

theorem function_through_points (a p q : ℝ) : 
  a > 0 →
  2^p / (2^p + a*p) = 6/5 →
  2^q / (2^q + a*q) = -1/5 →
  2^(p+q) = 16*p*q →
  a = 4 :=
by sorry

end function_through_points_l3065_306577


namespace quadratic_factorization_l3065_306524

theorem quadratic_factorization (x : ℝ) : x^2 - x - 42 = (x + 6) * (x - 7) := by
  sorry

end quadratic_factorization_l3065_306524


namespace geometric_sequence_third_term_l3065_306574

theorem geometric_sequence_third_term
  (a₁ : ℝ)
  (a₅ : ℝ)
  (h₁ : a₁ = 4)
  (h₂ : a₅ = 1296)
  (h₃ : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → ∃ r : ℝ, a₁ * r^(n-1) = a₁ * (a₅ / a₁)^((n-1)/4)) :
  ∃ a₃ : ℝ, a₃ = 36 ∧ a₃ = a₁ * (a₅ / a₁)^(1/2) := by
  sorry

end geometric_sequence_third_term_l3065_306574


namespace line_segment_endpoint_l3065_306560

def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

theorem line_segment_endpoint (endpoint1 midpoint : ℝ × ℝ) 
  (h : is_midpoint midpoint endpoint1 (1, 18)) : 
  endpoint1 = (5, 2) ∧ midpoint = (3, 10) → (1, 18) = (1, 18) := by
  sorry

end line_segment_endpoint_l3065_306560


namespace translation_not_equal_claim_l3065_306517

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the original function
def original (x : ℝ) : ℝ := f (-x)

-- Define the function after translation to the right by 1 unit
def translated (x : ℝ) : ℝ := f (-(x - 1))

-- Define the function claimed in the problem statement
def claimed (x : ℝ) : ℝ := f (-x - 1)

-- Theorem stating that the translated function is not equal to the claimed function
theorem translation_not_equal_claim : translated f ≠ claimed f := by sorry

end translation_not_equal_claim_l3065_306517


namespace multiplier_is_three_l3065_306593

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (26 - n) + 14) (h2 : n = 10) : 3 = 3 := by
  sorry

end multiplier_is_three_l3065_306593


namespace rectangular_box_existence_l3065_306584

theorem rectangular_box_existence : ∃ (a b c : ℕ), 
  a * b * c ≥ 1995 ∧ 2 * (a * b + b * c + a * c) = 958 := by
  sorry

end rectangular_box_existence_l3065_306584


namespace hyperbola_asymptotes_l3065_306502

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (a : ℝ) : Prop := hyperbola (Real.sqrt 13) 0 a

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a : ℝ) :
  right_focus a → ∀ x y, hyperbola x y a → asymptotes x y :=
sorry

end hyperbola_asymptotes_l3065_306502


namespace f_range_l3065_306536

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 2 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc (-12) 0 :=
by sorry

end f_range_l3065_306536


namespace calculate_expression_l3065_306597

theorem calculate_expression : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := by
  sorry

end calculate_expression_l3065_306597


namespace arithmetic_equalities_l3065_306599

theorem arithmetic_equalities :
  (96 * 98 * 189 = 81 * 343 * 2^6) ∧
  (12^18 = 27^6 * 16^9) ∧
  (25^28 * 0.008^19 ≠ 0.25) := by
sorry

end arithmetic_equalities_l3065_306599


namespace inscribed_circle_radius_l3065_306592

/-- The radius of a circle inscribed in a sector that is one-third of a circle --/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let r := (5 * Real.sqrt 3 - 5) / 2
  r > 0 ∧ r + r * Real.sqrt 3 = R :=
by sorry

end inscribed_circle_radius_l3065_306592


namespace student_line_count_l3065_306504

theorem student_line_count :
  ∀ (n : ℕ),
    n > 0 →
    (∃ (eunjung_pos yoojung_pos : ℕ),
      eunjung_pos = 5 ∧
      yoojung_pos = n ∧
      yoojung_pos - eunjung_pos - 1 = 8) →
    n = 14 := by
  sorry

end student_line_count_l3065_306504


namespace slope_angle_of_line_l3065_306513

/-- The slope angle of the line x + √3 * y - 5 = 0 is 150 degrees. -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ α : ℝ, α = 150 * π / 180 ∧ (Real.tan α = Real.sqrt 3) := by
  sorry

end slope_angle_of_line_l3065_306513


namespace remainder_7531_mod_11_l3065_306528

def digit_sum (n : ℕ) : ℕ := sorry

theorem remainder_7531_mod_11 :
  ∃ k : ℤ, 7531 = 11 * k + 5 :=
by
  have h1 : ∀ n : ℕ, ∃ k : ℤ, n = 11 * k + (digit_sum n % 11) := sorry
  sorry

end remainder_7531_mod_11_l3065_306528


namespace sum_of_parameters_l3065_306530

/-- Given two quadratic equations with solution sets M and N,
    prove that the sum of their parameters is 21 when their intersection is {2}. -/
theorem sum_of_parameters (p q : ℝ) : 
  (∃ M : Set ℝ, ∀ x ∈ M, x^2 - p*x + 6 = 0) →
  (∃ N : Set ℝ, ∀ x ∈ N, x^2 + 6*x - q = 0) →
  (∃ M N : Set ℝ, 
    (∀ x ∈ M, x^2 - p*x + 6 = 0) ∧ 
    (∀ x ∈ N, x^2 + 6*x - q = 0) ∧ 
    (M ∩ N = {2})) →
  p + q = 21 := by
  sorry

end sum_of_parameters_l3065_306530


namespace comparison_of_powers_l3065_306526

theorem comparison_of_powers : (2^40 : ℕ) < 3^28 ∧ (31^11 : ℕ) < 17^14 := by sorry

end comparison_of_powers_l3065_306526


namespace quadratic_roots_property_l3065_306571

theorem quadratic_roots_property (m : ℝ) (hm : m ≠ -1) :
  let f : ℝ → ℝ := λ x => (m + 1) * x^2 + 4 * m * x + m - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ < -1 ∨ x₂ < -1) := by
  sorry

end quadratic_roots_property_l3065_306571


namespace sin_monotone_decreasing_l3065_306541

theorem sin_monotone_decreasing (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 3 - 2 * x)
  ∀ x y, x ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         y ∈ Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) →
         x ≤ y → f y ≤ f x :=
by sorry

end sin_monotone_decreasing_l3065_306541


namespace readers_of_both_l3065_306558

theorem readers_of_both (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 150) 
  (h2 : science_fiction = 120) 
  (h3 : literary = 90) :
  science_fiction + literary - total = 60 := by
  sorry

end readers_of_both_l3065_306558


namespace unique_tangent_length_l3065_306579

theorem unique_tangent_length (m n t₁ : ℝ) : 
  (30 : ℝ) = m + n →
  t₁^2 = m * n →
  m ∈ Set.Ioo 0 30 →
  ∃ k : ℕ, m = 2 * k →
  ∃! t₁ : ℝ, t₁ > 0 ∧ t₁^2 = m * (30 - m) :=
sorry

end unique_tangent_length_l3065_306579


namespace article_cost_price_l3065_306575

/-- Given an article with marked price M and cost price C,
    prove that if 0.95M = 1.25C = 75, then C = 60. -/
theorem article_cost_price (M C : ℝ) (h : 0.95 * M = 1.25 * C ∧ 0.95 * M = 75) : C = 60 := by
  sorry

end article_cost_price_l3065_306575


namespace book_pages_proof_l3065_306564

/-- The number of pages Jack reads per day -/
def pages_per_day : ℕ := 23

/-- The number of pages Jack reads on the last day -/
def last_day_pages : ℕ := 9

/-- The total number of pages in the book -/
def total_pages : ℕ := 32

theorem book_pages_proof :
  ∃ (full_days : ℕ), total_pages = pages_per_day * full_days + last_day_pages :=
by sorry

end book_pages_proof_l3065_306564


namespace total_insect_legs_l3065_306522

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs per insect -/
def legs_per_insect : ℕ := 6

/-- Theorem: The total number of insect legs in the laboratory is 36 -/
theorem total_insect_legs : num_insects * legs_per_insect = 36 := by
  sorry

end total_insect_legs_l3065_306522


namespace unique_integer_pair_satisfying_equation_l3065_306552

theorem unique_integer_pair_satisfying_equation : 
  ∃! (m n : ℤ), m + 2*n = m*n + 2 := by
  sorry

end unique_integer_pair_satisfying_equation_l3065_306552


namespace unique_solution_iff_in_set_l3065_306545

/-- The set of real numbers m for which the equation 2√(1-m(x+2)) = x+4 has exactly one solution -/
def solution_set : Set ℝ :=
  {m : ℝ | m > -1/2 ∨ m = -1}

/-- The equation 2√(1-m(x+2)) = x+4 -/
def equation (m : ℝ) (x : ℝ) : Prop :=
  2 * Real.sqrt (1 - m * (x + 2)) = x + 4

theorem unique_solution_iff_in_set (m : ℝ) :
  (∃! x, equation m x) ↔ m ∈ solution_set :=
sorry

end unique_solution_iff_in_set_l3065_306545


namespace custom_mult_solution_l3065_306547

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that given the custom multiplication and the equation a * 7 = 16, a equals 32.5 -/
theorem custom_mult_solution :
  ∃ a : ℝ, custom_mult a 7 = 16 ∧ a = 32.5 := by sorry

end custom_mult_solution_l3065_306547


namespace rectangle_side_lengths_l3065_306576

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 := by
  sorry

end rectangle_side_lengths_l3065_306576


namespace power_of_two_plus_one_square_l3065_306588

theorem power_of_two_plus_one_square (m n : ℕ+) :
  2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ m = 3 ∧ n = 3 := by
  sorry

end power_of_two_plus_one_square_l3065_306588


namespace lisa_walking_speed_l3065_306519

/-- The number of meters Lisa walks per minute -/
def meters_per_minute (total_distance : ℕ) (days : ℕ) (hours_per_day : ℕ) (minutes_per_hour : ℕ) : ℚ :=
  (total_distance : ℚ) / (days * hours_per_day * minutes_per_hour)

/-- Proof that Lisa walks 10 meters per minute -/
theorem lisa_walking_speed :
  let total_distance := 1200
  let days := 2
  let hours_per_day := 1
  let minutes_per_hour := 60
  meters_per_minute total_distance days hours_per_day minutes_per_hour = 10 := by
  sorry

#eval meters_per_minute 1200 2 1 60

end lisa_walking_speed_l3065_306519


namespace inequality_proof_l3065_306589

theorem inequality_proof (n : ℕ) : 
  2 * Real.sqrt (n + 1 : ℝ) - 2 * Real.sqrt (n : ℝ) < 1 / Real.sqrt (n : ℝ) ∧ 
  1 / Real.sqrt (n : ℝ) < 2 * Real.sqrt (n : ℝ) - 2 * Real.sqrt ((n - 1) : ℝ) := by
  sorry

end inequality_proof_l3065_306589


namespace gcf_lcm_problem_l3065_306590

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 21) (LCM 10 15) = 3 := by
  sorry

end gcf_lcm_problem_l3065_306590


namespace shirts_sold_l3065_306544

/-- The number of shirts sold in a store -/
theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 :=
by sorry

end shirts_sold_l3065_306544


namespace emilys_glue_sticks_l3065_306527

theorem emilys_glue_sticks (total : ℕ) (sisters : ℕ) (emilys : ℕ) : 
  total = 13 → sisters = 7 → emilys = total - sisters → emilys = 6 :=
by sorry

end emilys_glue_sticks_l3065_306527


namespace fraction_power_six_l3065_306580

theorem fraction_power_six :
  (5 / 3 : ℚ) ^ 6 = 15625 / 729 := by sorry

end fraction_power_six_l3065_306580


namespace complex_exponential_sum_l3065_306507

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end complex_exponential_sum_l3065_306507


namespace rectangle_to_square_l3065_306595

/-- Given a rectangle with area 54 m², if one side is tripled and the other is halved to form a square, 
    the side length of the resulting square is 9 m. -/
theorem rectangle_to_square (a b : ℝ) (h1 : a * b = 54) (h2 : 3 * a = b / 2) : 
  3 * a = 9 ∧ b / 2 = 9 :=
by sorry

end rectangle_to_square_l3065_306595


namespace yoongi_calculation_l3065_306514

theorem yoongi_calculation : (30 + 5) / 7 = 5 := by
  sorry

end yoongi_calculation_l3065_306514


namespace cut_difference_l3065_306563

/-- The amount cut off the skirt in inches -/
def skirt_cut : ℝ := 0.75

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The difference between the amount cut off the skirt and the amount cut off the pants -/
theorem cut_difference : skirt_cut - pants_cut = 0.25 := by
  sorry

end cut_difference_l3065_306563


namespace spiral_stripe_length_l3065_306553

theorem spiral_stripe_length (c h : ℝ) (hc : c = 18) (hh : h = 8) :
  let stripe_length := Real.sqrt ((2 * c)^2 + h^2)
  stripe_length = Real.sqrt 1360 := by
  sorry

end spiral_stripe_length_l3065_306553


namespace cone_volume_relation_l3065_306500

/-- Represents a cone with given dimensions and properties -/
structure Cone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  l : ℝ  -- slant height
  d : ℝ  -- distance from center of base to slant height
  S : ℝ  -- lateral surface area
  V : ℝ  -- volume
  r_pos : 0 < r
  h_pos : 0 < h
  l_pos : 0 < l
  d_pos : 0 < d
  S_pos : 0 < S
  V_pos : 0 < V
  S_eq : S = π * r * l
  V_eq : V = (1/3) * π * r^2 * h

/-- The volume of a cone is one-third of the product of its lateral surface area and the distance from the center of the base to the slant height -/
theorem cone_volume_relation (c : Cone) : c.V = (1/3) * c.d * c.S := by
  sorry

end cone_volume_relation_l3065_306500


namespace rectangle_existence_theorem_l3065_306518

/-- Represents a rectangle with given side lengths -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Checks if a rectangle B exists with half the perimeter and area of rectangle A -/
def exists_half_rectangle (A : Rectangle) : Prop :=
  ∃ x : ℝ, x * ((A.a + A.b) / 2 - x) = A.a * A.b / 2

theorem rectangle_existence_theorem (A : Rectangle) :
  (A.a = 6 ∧ A.b = 1 → exists_half_rectangle A) ∧
  (A.a = 2 ∧ A.b = 1 → ¬exists_half_rectangle A) := by
  sorry

#check rectangle_existence_theorem

end rectangle_existence_theorem_l3065_306518


namespace number_of_boys_l3065_306585

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 150 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 60 := by
sorry

end number_of_boys_l3065_306585


namespace no_a_exists_for_union_range_of_a_for_intersection_l3065_306537

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | a*x^2 - 2*x + 8 = 0}

-- Theorem 1: There does not exist a real number 'a' such that A ∪ B = {0, 2, 4}
theorem no_a_exists_for_union : ¬ ∃ a : ℝ, A ∪ B a = {0, 2, 4} := by
  sorry

-- Theorem 2: The range of 'a' when A ∩ B = B is {0} ∪ (1/8, +∞)
theorem range_of_a_for_intersection (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a = 0 ∨ a > 1/8) := by
  sorry

end no_a_exists_for_union_range_of_a_for_intersection_l3065_306537


namespace constant_altitude_triangle_l3065_306535

/-- Given an equilateral triangle and a line through its center, prove the existence of a triangle
    with constant altitude --/
theorem constant_altitude_triangle (a : ℝ) (m : ℝ) :
  let A : ℝ × ℝ := (0, Real.sqrt 3 * a)
  let B : ℝ × ℝ := (-a, 0)
  let C : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let N : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let M : ℝ × ℝ := (-Real.sqrt 3 * a / (3 * m), 0)
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let BN := Real.sqrt ((N.1 - B.1)^2 + (N.2 - B.2)^2)
  let MN := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)
  ∃ (D E F : ℝ × ℝ),
    let h := Real.sqrt 6 * a / 3
    (E.1 - D.1)^2 + (E.2 - D.2)^2 = MN^2 ∧
    (F.1 - D.1)^2 + (F.2 - D.2)^2 = AM^2 ∧
    (F.1 - E.1)^2 + (F.2 - E.2)^2 = BN^2 ∧
    2 * (abs ((F.2 - E.2) * D.1 + (E.1 - F.1) * D.2 + (F.1 * E.2 - E.1 * F.2)) / Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)) = h :=
by
  sorry


end constant_altitude_triangle_l3065_306535


namespace ninth_grade_test_attendance_l3065_306561

theorem ninth_grade_test_attendance :
  let total_students : ℕ := 180
  let bombed_finals : ℕ := total_students / 4
  let remaining_students : ℕ := total_students - bombed_finals
  let passed_finals : ℕ := 70
  let less_than_d : ℕ := 20
  let took_test : ℕ := passed_finals + less_than_d
  let didnt_show_up : ℕ := remaining_students - took_test
  (didnt_show_up : ℚ) / remaining_students = 1 / 3 := by
  sorry

end ninth_grade_test_attendance_l3065_306561


namespace election_theorem_l3065_306515

/-- Represents a ballot with candidate names -/
structure Ballot where
  candidates : Finset String
  constraint : candidates.card = 10

/-- Represents a ballot box containing ballots -/
structure BallotBox where
  ballots : Set Ballot
  nonempty : ballots.Nonempty

/-- The election setup -/
structure Election where
  boxes : Fin 11 → BallotBox
  common_candidate : ∀ (selection : Fin 11 → Ballot), 
    (∀ i, selection i ∈ (boxes i).ballots) → 
    ∃ c, ∀ i, c ∈ (selection i).candidates

theorem election_theorem (e : Election) :
  ∃ i : Fin 11, ∃ c : String, ∀ b ∈ (e.boxes i).ballots, c ∈ b.candidates :=
sorry

end election_theorem_l3065_306515


namespace fourth_person_truthful_l3065_306539

/-- Represents a person who can be either a liar or truthful. -/
inductive Person
| Liar
| Truthful

/-- The statements made by each person. -/
def statement (p : Fin 4 → Person) : Prop :=
  (p 0 = Person.Liar ∧ p 1 = Person.Liar ∧ p 2 = Person.Liar ∧ p 3 = Person.Liar) ∨
  (∃! i, p i = Person.Liar) ∨
  (∃ i j, i ≠ j ∧ p i = Person.Liar ∧ p j = Person.Liar ∧ ∀ k, k ≠ i → k ≠ j → p k = Person.Truthful) ∨
  (p 3 = Person.Truthful)

/-- The main theorem stating that the fourth person must be truthful. -/
theorem fourth_person_truthful :
  ∀ p : Fin 4 → Person, statement p → p 3 = Person.Truthful :=
sorry

end fourth_person_truthful_l3065_306539


namespace min_transactions_to_identify_coins_l3065_306511

/-- Represents the set of coin values available -/
def CoinValues : Finset Nat := {1, 2, 5, 10, 20}

/-- The cost of one candy in florins -/
def CandyCost : Nat := 1

/-- Represents a vending machine transaction -/
structure Transaction where
  coin_inserted : Nat
  change_returned : Nat

/-- Function to determine if all coin values can be identified -/
def can_identify_all_coins (transactions : List Transaction) : Prop :=
  ∀ c ∈ CoinValues, ∃ t ∈ transactions, t.coin_inserted = c ∨ t.change_returned = c - CandyCost

/-- The main theorem stating that 4 is the minimum number of transactions required -/
theorem min_transactions_to_identify_coins :
  (∃ transactions : List Transaction, transactions.length = 4 ∧ can_identify_all_coins transactions) ∧
  (∀ transactions : List Transaction, transactions.length < 4 → ¬ can_identify_all_coins transactions) :=
sorry

end min_transactions_to_identify_coins_l3065_306511


namespace range_of_a_l3065_306566

-- Define the conditions p and q as functions of x and a
def p (x : ℝ) : Prop := abs (4 * x - 1) ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def neg_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ ∃ x, ¬(p x) ∧ (q x a)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, neg_p_necessary_not_sufficient a ↔ -1/2 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l3065_306566


namespace probability_of_three_common_books_l3065_306533

theorem probability_of_three_common_books :
  let total_books : ℕ := 12
  let books_per_student : ℕ := 6
  let common_books : ℕ := 3
  
  let total_outcomes : ℕ := (Nat.choose total_books books_per_student) ^ 2
  let successful_outcomes : ℕ := 
    (Nat.choose total_books common_books) * 
    (Nat.choose (total_books - common_books) (books_per_student - common_books)) * 
    (Nat.choose (total_books - books_per_student) (books_per_student - common_books))
  
  (successful_outcomes : ℚ) / total_outcomes = 5 / 23
  := by sorry

end probability_of_three_common_books_l3065_306533


namespace a_eq_one_sufficient_not_necessary_l3065_306573

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- Predicate that checks if f has only one zero for a given a -/
def has_only_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that a=1 is sufficient but not necessary for f to have only one zero -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → has_only_one_zero a) ∧
  ¬(∀ a : ℝ, has_only_one_zero a → a = 1) :=
sorry

end a_eq_one_sufficient_not_necessary_l3065_306573


namespace teachers_arrangements_count_l3065_306586

def num_students : ℕ := 5
def num_teachers : ℕ := 2

def arrangements (n_students : ℕ) (n_teachers : ℕ) : ℕ :=
  (Nat.factorial n_students) * (n_students - 1) * (Nat.factorial n_teachers)

theorem teachers_arrangements_count :
  arrangements num_students num_teachers = 960 := by
  sorry

end teachers_arrangements_count_l3065_306586


namespace complement_M_intersect_N_l3065_306581

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-3, -4} := by sorry

end complement_M_intersect_N_l3065_306581


namespace geometric_sequence_min_value_l3065_306546

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

def min_value (b : ℕ → ℝ) : ℝ :=
  5 * b 1 + 6 * b 2

theorem geometric_sequence_min_value :
  ∀ b : ℕ → ℝ, geometric_sequence b → b 0 = 2 →
  ∃ m : ℝ, m = min_value b ∧ m = -25/12 ∧ ∀ b' : ℕ → ℝ, geometric_sequence b' → b' 0 = 2 → min_value b' ≥ m :=
sorry

end geometric_sequence_min_value_l3065_306546


namespace class_size_is_36_l3065_306555

/-- The number of students in a class, given boat seating conditions. -/
def number_of_students (b : ℕ) : Prop :=
  ∃ n : ℕ,
    n = 6 * (b + 1) ∧
    n = 9 * (b - 1)

/-- Theorem stating that the number of students is 36. -/
theorem class_size_is_36 :
  ∃ b : ℕ, number_of_students b ∧ (6 * (b + 1) = 36) :=
sorry

end class_size_is_36_l3065_306555


namespace sunny_cakes_l3065_306532

/-- Given that Sunny gives away 2 cakes, puts 6 candles on each remaining cake,
    and uses a total of 36 candles, prove that she initially baked 8 cakes. -/
theorem sunny_cakes (cakes_given_away : ℕ) (candles_per_cake : ℕ) (total_candles : ℕ) :
  cakes_given_away = 2 →
  candles_per_cake = 6 →
  total_candles = 36 →
  cakes_given_away + (total_candles / candles_per_cake) = 8 := by
  sorry

end sunny_cakes_l3065_306532


namespace small_and_large_puzzle_cost_small_and_large_puzzle_cost_proof_l3065_306516

/-- The cost of a small puzzle and a large puzzle together is $23 -/
theorem small_and_large_puzzle_cost : ℝ → ℝ → Prop :=
  fun (small_cost large_cost : ℝ) ↦
    large_cost = 15 ∧
    large_cost + 3 * small_cost = 39 →
    small_cost + large_cost = 23

/-- Proof of the theorem -/
theorem small_and_large_puzzle_cost_proof :
  ∃ (small_cost large_cost : ℝ),
    small_and_large_puzzle_cost small_cost large_cost :=
by
  sorry

end small_and_large_puzzle_cost_small_and_large_puzzle_cost_proof_l3065_306516


namespace entree_percentage_is_80_percent_l3065_306540

/-- Calculates the percentage of total cost that went to entrees -/
def entree_percentage (total_cost appetizer_cost : ℚ) (num_appetizers : ℕ) : ℚ :=
  let appetizer_total := appetizer_cost * num_appetizers
  let entree_total := total_cost - appetizer_total
  (entree_total / total_cost) * 100

/-- Theorem stating that the percentage of total cost that went to entrees is 80% -/
theorem entree_percentage_is_80_percent :
  entree_percentage 50 5 2 = 80 := by
  sorry

end entree_percentage_is_80_percent_l3065_306540


namespace unfactorable_polynomial_l3065_306542

theorem unfactorable_polynomial (b c d : ℤ) (h : Odd (b * d + c * d)) :
  ¬ ∃ (p q r : ℤ), ∀ (x : ℤ), x^3 + b*x^2 + c*x + d = (x + p) * (x^2 + q*x + r) :=
sorry

end unfactorable_polynomial_l3065_306542


namespace rachel_books_total_l3065_306505

theorem rachel_books_total (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 9)
  (h2 : mystery_shelves = 6)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 72 :=
by sorry

end rachel_books_total_l3065_306505


namespace factorization_3m_squared_minus_12m_l3065_306587

theorem factorization_3m_squared_minus_12m (m : ℝ) : 3 * m^2 - 12 * m = 3 * m * (m - 4) := by
  sorry

end factorization_3m_squared_minus_12m_l3065_306587


namespace max_value_yzx_l3065_306582

theorem max_value_yzx (x y z : ℝ) 
  (h1 : x^2 + z^2 = 1) 
  (h2 : y^2 + 2*y*(x + z) = 6) : 
  ∃ (M : ℝ), M = Real.sqrt 7 ∧ ∀ (x' y' z' : ℝ), 
    x'^2 + z'^2 = 1 → y'^2 + 2*y'*(x' + z') = 6 → 
    y'*(z' - x') ≤ M :=
sorry

end max_value_yzx_l3065_306582


namespace function_roots_imply_a_range_l3065_306557

/-- The function f(x) = 2ln(x) - x^2 + a has two roots in [1/e, e] iff a ∈ (1, 2 + 1/e^2] -/
theorem function_roots_imply_a_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.log x - x^2 + a) →
  (∃ x y, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ f x = 0 ∧ f y = 0) →
  a ∈ Set.Ioo 1 (2 + 1 / (Real.exp 1)^2) :=
by sorry

end function_roots_imply_a_range_l3065_306557


namespace x_minus_y_value_l3065_306520

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 5) : 
  x - y = 73 / 13 := by
sorry

end x_minus_y_value_l3065_306520


namespace sum_of_integers_l3065_306551

theorem sum_of_integers (x y : ℤ) : 
  3 * x + 2 * y = 115 → (x = 25 ∨ y = 25) → (x = 20 ∨ y = 20) := by
  sorry

end sum_of_integers_l3065_306551


namespace min_value_quadratic_l3065_306550

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 45 / 8 := by
  sorry

end min_value_quadratic_l3065_306550


namespace positive_real_solution_l3065_306510

theorem positive_real_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d)/(b + 2*c + d) + (b^2 - c*a)/(c + 2*d + a) + 
  (c^2 - d*b)/(d + 2*a + b) + (d^2 - a*c)/(a + 2*b + c) = 0 →
  a = c ∧ b = d := by
sorry

end positive_real_solution_l3065_306510


namespace least_four_digit_divisible_by_digits_l3065_306594

/-- A function that checks if a number is a four-digit positive integer with all different digits -/
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10)

/-- A function that checks if a number is divisible by all of its digits -/
def divisible_by_digits (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 4 → (n % ((n / 10^i) % 10) = 0 ∨ (n / 10^i) % 10 = 0)

/-- The main theorem stating that 1236 is the least number satisfying the conditions -/
theorem least_four_digit_divisible_by_digits :
  is_valid_number 1236 ∧ 
  divisible_by_digits 1236 ∧
  (∀ m : ℕ, m < 1236 → ¬(is_valid_number m ∧ divisible_by_digits m)) :=
by sorry

end least_four_digit_divisible_by_digits_l3065_306594


namespace problem_statements_l3065_306567

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab - a - 2*b = 0 → a + 2*b ≥ 8) ∧
  (a + b = 1 → Real.sqrt (2*a + 4) + Real.sqrt (b + 1) ≤ 2 * Real.sqrt 3) ∧
  (1 / (a + 1) + 1 / (b + 2) = 1 / 3 → a*b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end problem_statements_l3065_306567


namespace normal_distribution_symmetry_l3065_306570

/-- Represents a normal distribution with mean μ and standard deviation σ -/
noncomputable def NormalDistribution (μ σ : ℝ) : Type :=
  ℝ → ℝ

/-- The probability that a random variable X from a normal distribution
    falls within the interval [a, b] -/
noncomputable def prob_between (X : NormalDistribution μ σ) (a b : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable X from a normal distribution
    is greater than or equal to a given value -/
noncomputable def prob_ge (X : NormalDistribution μ σ) (a : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (X : NormalDistribution 100 σ) 
  (h : prob_between X 80 120 = 3/4) : 
  prob_ge X 120 = 1/8 := by
  sorry

end normal_distribution_symmetry_l3065_306570
