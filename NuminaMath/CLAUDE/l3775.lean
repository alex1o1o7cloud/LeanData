import Mathlib

namespace roots_equation_l3775_377541

theorem roots_equation (n r s c d : ℝ) : 
  (c^2 - n*c + 6 = 0) →
  (d^2 - n*d + 6 = 0) →
  ((c^2 + 1/d)^2 - r*(c^2 + 1/d) + s = 0) →
  ((d^2 + 1/c)^2 - r*(d^2 + 1/c) + s = 0) →
  s = n + 217/6 := by
sorry

end roots_equation_l3775_377541


namespace function_inequality_l3775_377598

theorem function_inequality (a b : ℝ) (h_a : a > 0) : 
  (∃ x : ℝ, x > 0 ∧ Real.log x - a * x - b ≥ 0) → a * b ≤ Real.exp (-2) := by
  sorry

end function_inequality_l3775_377598


namespace collinear_points_x_value_l3775_377506

-- Define the points
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C : ℝ → ℝ × ℝ := λ x => (5, x)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Theorem statement
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end collinear_points_x_value_l3775_377506


namespace net_calorie_deficit_l3775_377572

/-- Calculates the net calorie deficit for a round trip walk and candy bar consumption. -/
theorem net_calorie_deficit
  (distance : ℝ)
  (calorie_burn_rate : ℝ)
  (candy_bar_calories : ℝ)
  (h1 : distance = 3)
  (h2 : calorie_burn_rate = 150)
  (h3 : candy_bar_calories = 200) :
  distance * calorie_burn_rate - candy_bar_calories = 250 :=
by sorry

end net_calorie_deficit_l3775_377572


namespace geometric_sequence_sum_l3775_377551

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_sum 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 2 * a 5 = -3/4) 
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) : 
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end geometric_sequence_sum_l3775_377551


namespace all_triangles_present_l3775_377511

/-- A permissible triangle with angles represented as integers -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (all_pos : 0 < a ∧ 0 < b ∧ 0 < c)

/-- The set of all permissible triangles for a given prime p -/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function representing the division process -/
def DivideTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles after the division process is complete -/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem -/
theorem all_triangles_present (p : ℕ) (hp : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end all_triangles_present_l3775_377511


namespace sunflower_seed_contest_l3775_377500

/-- The total number of seeds eaten by five players in a sunflower seed eating contest -/
def total_seeds (player1 player2 player3 player4 player5 : ℕ) : ℕ :=
  player1 + player2 + player3 + player4 + player5

/-- Theorem stating the total number of seeds eaten by the five players -/
theorem sunflower_seed_contest : ∃ (player1 player2 player3 player4 player5 : ℕ),
  player1 = 78 ∧
  player2 = 53 ∧
  player3 = player2 + 30 ∧
  player4 = 2 * player3 ∧
  player5 = (player1 + player2 + player3 + player4) / 4 ∧
  total_seeds player1 player2 player3 player4 player5 = 475 := by
  sorry

end sunflower_seed_contest_l3775_377500


namespace expression_equals_one_l3775_377514

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b - c = 0) :
  (a^2 * b^2) / ((a^2 + b*c) * (b^2 + a*c)) +
  (a^2 * c^2) / ((a^2 + b*c) * (c^2 + a*b)) +
  (b^2 * c^2) / ((b^2 + a*c) * (c^2 + a*b)) = 1 := by
sorry

end expression_equals_one_l3775_377514


namespace z_pure_imaginary_iff_a_eq_neg_two_l3775_377591

-- Define the complex number z as a function of real number a
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

-- Define what it means for a complex number to be purely imaginary
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- Theorem statement
theorem z_pure_imaginary_iff_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) ↔ a = -2 :=
by sorry

end z_pure_imaginary_iff_a_eq_neg_two_l3775_377591


namespace c_is_largest_l3775_377531

-- Define the numbers as real numbers
def a : ℝ := 7.25678
def b : ℝ := 7.256777777777777 -- Approximation of 7.256̄7
def c : ℝ := 7.257676767676767 -- Approximation of 7.25̄76
def d : ℝ := 7.275675675675675 -- Approximation of 7.2̄756
def e : ℝ := 7.275627562756275 -- Approximation of 7.̄2756

-- Theorem stating that c (7.25̄76) is the largest
theorem c_is_largest : 
  c > a ∧ c > b ∧ c > d ∧ c > e :=
sorry

end c_is_largest_l3775_377531


namespace sqrt_16_minus_2_squared_equals_zero_l3775_377567

theorem sqrt_16_minus_2_squared_equals_zero : 
  Real.sqrt 16 - 2^2 = 0 := by sorry

end sqrt_16_minus_2_squared_equals_zero_l3775_377567


namespace complex_modulus_l3775_377513

theorem complex_modulus (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end complex_modulus_l3775_377513


namespace digit_sum_in_multiplication_l3775_377557

theorem digit_sum_in_multiplication (c d : ℕ) : 
  c < 10 → d < 10 → 
  (30 + c) * (10 * d + 5) = 185 →
  5 * c = 15 →
  c + d = 11 := by
sorry

end digit_sum_in_multiplication_l3775_377557


namespace frustum_cone_volume_l3775_377574

/-- Given a frustum of a cone with volume 78 and one base area 9 times the other,
    the volume of the cone that cuts this frustum is 81. -/
theorem frustum_cone_volume (r R : ℝ) (h1 : r > 0) (h2 : R > 0) : 
  (π * (R^2 + r^2 + R*r) * (R - r) / 3 = 78) →
  (π * R^2 = 9 * π * r^2) →
  (π * R^3 / 3 = 81) := by
  sorry

end frustum_cone_volume_l3775_377574


namespace fraction_equality_l3775_377593

theorem fraction_equality : (8 : ℚ) / (5 * 48) = 0.8 / (5 * 0.48) := by
  sorry

end fraction_equality_l3775_377593


namespace three_number_sum_l3775_377515

theorem three_number_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  b = 10 → 
  a + b + c = -15 := by
  sorry

end three_number_sum_l3775_377515


namespace set_operations_l3775_377503

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∪ (B ∩ C) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}) ∧
  (A ∩ (A \ (B ∩ C)) = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 5, 6}) := by
  sorry

end set_operations_l3775_377503


namespace sequence_difference_sum_l3775_377545

-- Define the arithmetic sequences
def seq1 : List Nat := List.range 93 |>.map (fun i => i + 1981)
def seq2 : List Nat := List.range 93 |>.map (fun i => i + 201)

-- Define the sum of each sequence
def sum1 : Nat := seq1.sum
def sum2 : Nat := seq2.sum

-- Theorem statement
theorem sequence_difference_sum : sum1 - sum2 = 165540 := by
  sorry

end sequence_difference_sum_l3775_377545


namespace card_difference_l3775_377538

theorem card_difference (heike anton ann : ℕ) : 
  anton = 3 * heike →
  ann = 6 * heike →
  ann = 60 →
  ann - anton = 30 := by
sorry

end card_difference_l3775_377538


namespace statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l3775_377534

-- Statement A
theorem statement_a_is_false : ∃ (a b c : ℝ), a > b ∧ c < 0 ∧ a * c ≤ b * c :=
  sorry

-- Statement B
theorem statement_b_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (2 * x * y) / (x + y) < Real.sqrt (x * y) :=
  sorry

-- Statement C
theorem statement_c_is_true : ∀ (s : ℝ), s > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s → 
  x * y ≤ (s / 2) * (s / 2) :=
  sorry

-- Statement D
theorem statement_d_is_true : ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y → 
  (x^2 + y^2) / 2 > ((x + y) / 2)^2 :=
  sorry

-- Statement E
theorem statement_e_is_true : ∀ (p : ℝ), p > 0 → 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p → 
  x + y ≥ 2 * Real.sqrt p :=
  sorry

end statement_a_is_false_statement_b_is_true_statement_c_is_true_statement_d_is_true_statement_e_is_true_l3775_377534


namespace cookie_number_proof_l3775_377532

/-- The smallest positive integer satisfying the given conditions -/
def smallest_cookie_number : ℕ := 2549

/-- Proof that the smallest_cookie_number satisfies all conditions -/
theorem cookie_number_proof :
  smallest_cookie_number % 6 = 5 ∧
  smallest_cookie_number % 8 = 6 ∧
  smallest_cookie_number % 10 = 9 ∧
  ∃ k : ℕ, k * k = smallest_cookie_number ∧
  ∀ n : ℕ, n > 0 ∧ n < smallest_cookie_number →
    ¬(n % 6 = 5 ∧ n % 8 = 6 ∧ n % 10 = 9 ∧ ∃ m : ℕ, m * m = n) :=
by sorry

end cookie_number_proof_l3775_377532


namespace system_is_linear_l3775_377573

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y - c

/-- A system of two linear equations --/
def IsSystemOfTwoLinearEquations (f g : ℝ → ℝ → ℝ) : Prop :=
  IsLinearEquation f ∧ IsLinearEquation g

/-- The given system of equations --/
def f (x y : ℝ) : ℝ := 4 * x - y - 1
def g (x y : ℝ) : ℝ := y - 2 * x - 3

theorem system_is_linear : IsSystemOfTwoLinearEquations f g := by
  sorry

end system_is_linear_l3775_377573


namespace unique_stutterer_square_l3775_377583

/-- A function that checks if a number is a stutterer (first two digits are the same and last two digits are the same) --/
def is_stutterer (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

/-- The theorem stating that 7744 is the only four-digit stutterer number that is a perfect square --/
theorem unique_stutterer_square : ∀ n : ℕ, 
  is_stutterer n ∧ ∃ k : ℕ, n = k^2 ↔ n = 7744 :=
sorry

end unique_stutterer_square_l3775_377583


namespace g_composition_of_3_l3775_377561

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_composition_of_3 : g (g (g 3)) = 241 := by
  sorry

end g_composition_of_3_l3775_377561


namespace percent_change_condition_l3775_377577

theorem percent_change_condition (a b r N : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < r ∧ 0 < N ∧ r < 50 →
  (N * (1 + a / 100) * (1 - b / 100) ≤ N * (1 + r / 100) ↔ a - b - a * b / 100 ≤ r) :=
by sorry

end percent_change_condition_l3775_377577


namespace y_derivative_l3775_377529

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * (Real.sin x)^3) - 1 / (Real.sin x) + (1/2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem y_derivative (x : ℝ) (hx : Real.cos x ≠ 0) (hsx : Real.sin x ≠ 0) : 
  deriv y x = 1 / (Real.cos x * (Real.sin x)^4) :=
sorry

end y_derivative_l3775_377529


namespace select_five_from_ten_l3775_377527

theorem select_five_from_ten : Nat.choose 10 5 = 2520 := by
  sorry

end select_five_from_ten_l3775_377527


namespace board_number_remainder_l3775_377555

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by sorry

end board_number_remainder_l3775_377555


namespace integer_roots_of_polynomial_l3775_377578

def polynomial (x : ℤ) : ℤ := x^4 + 2*x^3 - x^2 + 3*x - 30

def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end integer_roots_of_polynomial_l3775_377578


namespace tan_two_alpha_l3775_377507

theorem tan_two_alpha (α β : ℝ) (h1 : Real.tan (α - β) = -3/2) (h2 : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := by
  sorry

end tan_two_alpha_l3775_377507


namespace triangle_side_length_l3775_377562

theorem triangle_side_length (b : ℝ) (B : ℝ) (A : ℝ) (a : ℝ) :
  b = 5 → B = π / 4 → Real.sin A = 1 / 3 → a = 5 * Real.sqrt 2 / 3 := by
  sorry

end triangle_side_length_l3775_377562


namespace max_correct_is_38_l3775_377512

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest. -/
def max_correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 38. -/
theorem max_correct_is_38 :
  let contest := MathContest.mk 60 5 0 (-2) 150
  max_correct_answers contest = 38 := by
  sorry

end max_correct_is_38_l3775_377512


namespace smallest_x_value_l3775_377599

theorem smallest_x_value (x : ℝ) : 
  x ≠ 9 → x ≠ -7 → (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 7) → 
  ∃ (y : ℝ), y = -8 ∧ (y^2 - 5*y - 84) / (y - 9) = 4 / (y + 7) ∧ 
  ∀ (z : ℝ), z ≠ 9 → z ≠ -7 → (z^2 - 5*z - 84) / (z - 9) = 4 / (z + 7) → y ≤ z :=
by sorry

end smallest_x_value_l3775_377599


namespace compute_expression_l3775_377548

/-- Operation Δ: a Δ b = a × 100...00 (b zeros) + b -/
def delta (a b : ℕ) : ℕ := a * (10^b) + b

/-- Operation □: a □ b = a × 10 + b -/
def square (a b : ℕ) : ℕ := a * 10 + b

/-- Theorem: 2018 □ (123 Δ 4) = 1250184 -/
theorem compute_expression : square 2018 (delta 123 4) = 1250184 := by sorry

end compute_expression_l3775_377548


namespace annas_earnings_is_96_l3775_377585

/-- Calculates Anna's earnings from selling cupcakes given the number of trays, cupcakes per tray, price per cupcake, and fraction sold. -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (fraction_sold : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * fraction_sold * price_per_cupcake

/-- Theorem stating that Anna's earnings are $96 given the specific conditions. -/
theorem annas_earnings_is_96 :
  annas_earnings 4 20 2 (3/5) = 96 := by
  sorry

end annas_earnings_is_96_l3775_377585


namespace thirtieth_triangular_and_sum_l3775_377556

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_and_sum :
  (triangular_number 30 = 465) ∧
  (triangular_number 30 + triangular_number 31 = 961) := by
  sorry

end thirtieth_triangular_and_sum_l3775_377556


namespace round_trip_speed_ratio_l3775_377588

/-- Proves that given a round trip with specific conditions, the ratio of return speed to outward speed is 2 --/
theorem round_trip_speed_ratio
  (distance : ℝ)
  (total_time : ℝ)
  (return_speed : ℝ)
  (h_distance : distance = 35)
  (h_total_time : total_time = 6)
  (h_return_speed : return_speed = 17.5)
  : (return_speed / ((2 * distance) / total_time - return_speed)) = 2 := by
  sorry

#check round_trip_speed_ratio

end round_trip_speed_ratio_l3775_377588


namespace maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l3775_377564

/-- Given Maddie's TV watching schedule, prove she watched 105 minutes over the weekend -/
theorem maddie_weekend_watching_time : ℕ → Prop :=
  λ weekend_minutes : ℕ =>
    let total_episodes : ℕ := 8
    let minutes_per_episode : ℕ := 44
    let monday_minutes : ℕ := 138
    let thursday_minutes : ℕ := 21
    let friday_episodes : ℕ := 2

    let total_minutes : ℕ := total_episodes * minutes_per_episode
    let weekday_minutes : ℕ := monday_minutes + thursday_minutes + (friday_episodes * minutes_per_episode)

    weekend_minutes = total_minutes - weekday_minutes ∧ weekend_minutes = 105

/-- Proof of the theorem -/
theorem maddie_weekend_watching_time_proof : maddie_weekend_watching_time 105 := by
  sorry

end maddie_weekend_watching_time_maddie_weekend_watching_time_proof_l3775_377564


namespace suzanna_textbook_pages_l3775_377526

/-- Calculate the total number of pages in Suzanna's textbooks --/
theorem suzanna_textbook_pages : 
  let history_pages : ℕ := 160
  let geography_pages : ℕ := history_pages + 70
  let math_pages : ℕ := (history_pages + geography_pages) / 2
  let science_pages : ℕ := 2 * history_pages
  history_pages + geography_pages + math_pages + science_pages = 905 :=
by sorry

end suzanna_textbook_pages_l3775_377526


namespace angle_on_ray_l3775_377558

/-- Given an angle α where its initial side coincides with the non-negative half-axis of the x-axis
    and its terminal side lies on the ray 4x - 3y = 0 (x ≤ 0), cos α - sin α = 1/5 -/
theorem angle_on_ray (α : Real) : 
  (∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin α * Real.sqrt (x^2 + y^2)) → 
  Real.cos α - Real.sin α = 1/5 := by
sorry

end angle_on_ray_l3775_377558


namespace product_mod_seven_l3775_377528

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end product_mod_seven_l3775_377528


namespace quadratic_function_max_value_l3775_377579

/-- Given a quadratic function f(x) = ax^2 - 4x + c where a ≠ 0,
    with range [0, +∞) and f(1) ≤ 4, the maximum value of
    u = a/(c^2+4) + c/(a^2+4) is 7/4. -/
theorem quadratic_function_max_value (a c : ℝ) (h1 : a ≠ 0) :
  let f := fun x => a * x^2 - 4 * x + c
  (∀ y, y ∈ Set.range f → y ≥ 0) →
  (f 1 ≤ 4) →
  (∃ u : ℝ, u = a / (c^2 + 4) + c / (a^2 + 4) ∧
    u ≤ 7/4 ∧
    ∀ v, v = a / (c^2 + 4) + c / (a^2 + 4) → v ≤ u) :=
by sorry

end quadratic_function_max_value_l3775_377579


namespace a_greater_than_b_l3775_377594

theorem a_greater_than_b (A B : ℝ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : A * 4 = B * 5) : A > B := by
  sorry

end a_greater_than_b_l3775_377594


namespace parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l3775_377560

/-- A parabola with vertex at the origin and focus on the positive y-axis -/
structure UpwardParabola where
  focus : ℝ
  focus_positive : 0 < focus

/-- The equation of an upward parabola given its focus -/
def parabola_equation (p : UpwardParabola) (x y : ℝ) : Prop :=
  x^2 = 2 * p.focus * y

/-- A line passing through two points on a parabola -/
structure IntersectingLine (p : UpwardParabola) where
  slope : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = slope * x₁ + p.focus ∧
    y₂ = slope * x₂ + p.focus

/-- The main theorem -/
theorem parabola_equation_from_dot_product
  (p : UpwardParabola)
  (l : IntersectingLine p)
  (h : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_equation p x₁ y₁ ∧
    parabola_equation p x₂ y₂ ∧
    y₁ = l.slope * x₁ + p.focus ∧
    y₂ = l.slope * x₂ + p.focus ∧
    x₁ * x₂ + y₁ * y₂ = -12) :
  p.focus = 4 :=
sorry

/-- The equation of the parabola is x² = 8y -/
theorem parabola_equation_is_x_squared_eq_8y
  (p : UpwardParabola)
  (h : p.focus = 4) :
  ∀ x y, parabola_equation p x y ↔ x^2 = 8*y :=
sorry

end parabola_equation_from_dot_product_parabola_equation_is_x_squared_eq_8y_l3775_377560


namespace rachel_brownies_l3775_377549

def brownies_baked (brought_to_school left_at_home : ℕ) : ℕ :=
  brought_to_school + left_at_home

theorem rachel_brownies : 
  brownies_baked 16 24 = 40 := by
  sorry

end rachel_brownies_l3775_377549


namespace cos_12_18_minus_sin_12_18_l3775_377554

theorem cos_12_18_minus_sin_12_18 :
  Real.cos (12 * π / 180) * Real.cos (18 * π / 180) - 
  Real.sin (12 * π / 180) * Real.sin (18 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end cos_12_18_minus_sin_12_18_l3775_377554


namespace exam_maximum_marks_l3775_377525

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (paul_marks : ℕ) (failing_margin : ℕ),
    paul_marks = 50 →
    failing_margin = 10 →
    paul_marks + failing_margin = max_marks / 2 →
    max_marks = 120 := by
  sorry

end exam_maximum_marks_l3775_377525


namespace arithmetic_sequence_150th_term_l3775_377570

/-- 
Given an arithmetic sequence where:
  - The first term is 2
  - The common difference is 4
Prove that the 150th term of this sequence is 598
-/
theorem arithmetic_sequence_150th_term : 
  ∀ (a : ℕ → ℕ), 
  (a 1 = 2) →  -- First term is 2
  (∀ n, a (n + 1) = a n + 4) →  -- Common difference is 4
  a 150 = 598 := by
sorry

end arithmetic_sequence_150th_term_l3775_377570


namespace star_associative_l3775_377518

variable {U : Type*}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associative (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end star_associative_l3775_377518


namespace sum_of_coefficients_l3775_377582

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end sum_of_coefficients_l3775_377582


namespace complete_square_factorization_l3775_377587

theorem complete_square_factorization (a : ℝ) :
  a^2 - 6*a + 8 = (a - 4) * (a - 2) := by sorry

end complete_square_factorization_l3775_377587


namespace laundry_time_proof_l3775_377505

/-- Proves that the time to wash one load of laundry is 45 minutes -/
theorem laundry_time_proof (wash_time : ℕ) : 
  (2 * wash_time + 75 = 165) → wash_time = 45 := by
  sorry

end laundry_time_proof_l3775_377505


namespace bus_car_length_ratio_l3775_377533

theorem bus_car_length_ratio : 
  ∀ (red_bus_length orange_car_length yellow_bus_length : ℝ),
  red_bus_length = 4 * orange_car_length →
  red_bus_length = 48 →
  yellow_bus_length = red_bus_length - 6 →
  yellow_bus_length / orange_car_length = 7 / 2 := by
sorry

end bus_car_length_ratio_l3775_377533


namespace square_field_diagonal_l3775_377552

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 50 → diagonal = 10 := by sorry

end square_field_diagonal_l3775_377552


namespace max_parts_three_planes_l3775_377544

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition would go here

/-- The maximum number of parts that three planes can divide 3D space into -/
theorem max_parts_three_planes :
  ∃ (planes : List Plane3D), planes.length = 3 ∧ 
  ∀ (other_planes : List Plane3D), other_planes.length = 3 →
  num_parts other_planes ≤ num_parts planes ∧ num_parts planes = 8 :=
sorry

end max_parts_three_planes_l3775_377544


namespace cubic_sum_ratio_l3775_377546

theorem cubic_sum_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^6 + y^6 + z^6) / (x*y*z * (x*y + x*z + y*z)) = 6 := by
  sorry

end cubic_sum_ratio_l3775_377546


namespace expression_simplification_and_evaluation_l3775_377569

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - 1 / (x + 1)) * ((x^2 - 1) / x) = Real.sqrt 2 := by
  sorry

end expression_simplification_and_evaluation_l3775_377569


namespace evaluate_expression_l3775_377537

theorem evaluate_expression : 2000^3 - 1999 * 2000^2 - 1999^2 * 2000 + 1999^3 = 3999 := by
  sorry

end evaluate_expression_l3775_377537


namespace soccer_expansion_l3775_377517

/-- The total number of kids playing soccer after expansion -/
def total_kids (initial : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial + initial * friends_per_kid

/-- Theorem stating that with 14 initial kids and 3 friends per kid, the total is 56 -/
theorem soccer_expansion : total_kids 14 3 = 56 := by
  sorry

end soccer_expansion_l3775_377517


namespace special_function_value_l3775_377523

/-- A function satisfying certain properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) = f (1 - x)) ∧ 
  (∀ x, f (x + 2) = f (x + 1) - f x) ∧
  (f 1 = 1/2)

/-- Theorem stating that for any function satisfying the special properties, f(2024) = 1/4 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2024 = 1/4 := by
  sorry

end special_function_value_l3775_377523


namespace truck_filling_time_truck_filling_time_proof_l3775_377535

/-- The time taken to fill a truck with stone blocks given specific worker rates and capacity -/
theorem truck_filling_time : ℕ :=
  let truck_capacity : ℕ := 6000
  let stella_initial_rate : ℕ := 250
  let twinkle_initial_rate : ℕ := 200
  let stella_changed_rate : ℕ := 220
  let twinkle_changed_rate : ℕ := 230
  let additional_workers_count : ℕ := 6
  let additional_workers_initial_rate1 : ℕ := 300
  let additional_workers_initial_rate2 : ℕ := 180
  let additional_workers_changed_rate1 : ℕ := 280
  let additional_workers_changed_rate2 : ℕ := 190
  let initial_period : ℕ := 2
  let second_period : ℕ := 4
  let additional_workers_initial_period : ℕ := 1

  8

theorem truck_filling_time_proof : truck_filling_time = 8 := by
  sorry

end truck_filling_time_truck_filling_time_proof_l3775_377535


namespace greatest_three_digit_base_8_divisible_by_7_l3775_377581

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def is_three_digit_base_8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_three_digit_base_8_divisible_by_7 :
  ∀ n : Nat, is_three_digit_base_8 n → (base_8_to_decimal n) % 7 = 0 →
  n ≤ 777 :=
by sorry

end greatest_three_digit_base_8_divisible_by_7_l3775_377581


namespace sum_of_threes_plus_product_of_fours_l3775_377542

theorem sum_of_threes_plus_product_of_fours (m n : ℕ) :
  (List.replicate m 3).sum + (List.replicate n 4).prod = 3 * m + 4^n := by
  sorry

end sum_of_threes_plus_product_of_fours_l3775_377542


namespace cricket_solution_l3775_377584

def cricket_problem (initial_average : ℝ) (runs_10th_innings : ℕ) : Prop :=
  let total_runs_9_innings := 9 * initial_average
  let total_runs_10_innings := total_runs_9_innings + runs_10th_innings
  let new_average := total_runs_10_innings / 10
  (new_average = initial_average + 8) ∧ (new_average = 128)

theorem cricket_solution :
  ∀ initial_average : ℝ,
  ∃ runs_10th_innings : ℕ,
  cricket_problem initial_average runs_10th_innings ∧
  runs_10th_innings = 200 :=
by sorry

end cricket_solution_l3775_377584


namespace experience_ratio_l3775_377586

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.roger + 8 = 50 ∧
  e.peter = 12 ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

theorem experience_ratio (e : Experience) 
  (h : satisfiesConditions e) : e.tom = e.robert :=
sorry

end experience_ratio_l3775_377586


namespace inverse_trig_sum_l3775_377536

theorem inverse_trig_sum : 
  Real.arcsin (-1/2) + Real.arccos (-Real.sqrt 3/2) + Real.arctan (-Real.sqrt 3) = π/3 := by
  sorry

end inverse_trig_sum_l3775_377536


namespace digit_product_puzzle_l3775_377539

theorem digit_product_puzzle :
  ∀ (A B C D : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (10 * A + B) * (10 * C + B) = 111 * D →
    10 * A + B < 10 * C + B →
    A = 2 ∧ B = 7 ∧ C = 3 ∧ D = 9 := by
  sorry

end digit_product_puzzle_l3775_377539


namespace power_function_even_l3775_377553

-- Define the power function f
def f (x : ℝ) : ℝ := x^(2/3)

-- Theorem statement
theorem power_function_even : 
  (f 8 = 4) → (∀ x : ℝ, f (-x) = f x) :=
by
  sorry

end power_function_even_l3775_377553


namespace male_wage_is_35_l3775_377501

/-- Represents the daily wage structure and worker composition of a building contractor -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a male worker given the contractor's data -/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 35 -/
theorem male_wage_is_35 (data : ContractorData) 
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  male_wage data = 35 := by
  sorry

#eval male_wage { 
  male_workers := 20, 
  female_workers := 15, 
  child_workers := 5, 
  female_wage := 20, 
  child_wage := 8, 
  average_wage := 26 
}

end male_wage_is_35_l3775_377501


namespace arithmetic_geometric_ratio_l3775_377508

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 5) (a 9) (a 15)) :
  a 9 / a 5 = 3 / 2 :=
sorry

end arithmetic_geometric_ratio_l3775_377508


namespace steve_reading_time_l3775_377502

/-- Represents Steve's daily reading schedule in pages -/
def daily_reading : Fin 7 → ℕ
  | 0 => 100  -- Monday
  | 1 => 150  -- Tuesday
  | 2 => 100  -- Wednesday
  | 3 => 150  -- Thursday
  | 4 => 100  -- Friday
  | 5 => 50   -- Saturday
  | 6 => 0    -- Sunday

/-- The total number of pages in the book -/
def book_length : ℕ := 2100

/-- Calculate the total pages read in a week -/
def pages_per_week : ℕ := (List.range 7).map daily_reading |>.sum

/-- The number of weeks needed to read the book -/
def weeks_to_read : ℕ := (book_length + pages_per_week - 1) / pages_per_week

theorem steve_reading_time :
  weeks_to_read = 4 := by sorry

end steve_reading_time_l3775_377502


namespace election_votes_l3775_377524

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) :
  total_votes = 7500 →
  invalid_percent = 1/5 →
  winner_percent = 11/20 →
  ∃ (other_candidate_votes : ℕ), other_candidate_votes = 2700 := by
  sorry

end election_votes_l3775_377524


namespace amy_treasures_first_level_l3775_377597

def points_per_treasure : ℕ := 4
def treasures_second_level : ℕ := 2
def total_score : ℕ := 32

def treasures_first_level : ℕ := (total_score - points_per_treasure * treasures_second_level) / points_per_treasure

theorem amy_treasures_first_level : treasures_first_level = 6 := by
  sorry

end amy_treasures_first_level_l3775_377597


namespace tucker_tissues_left_l3775_377580

/-- The number of tissues left after buying boxes and using some. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given 160 tissues per box, 3 boxes bought, and 210 tissues used, 270 tissues are left. -/
theorem tucker_tissues_left :
  tissues_left 160 3 210 = 270 := by
  sorry

end tucker_tissues_left_l3775_377580


namespace dress_design_combinations_l3775_377571

theorem dress_design_combinations (num_colors num_patterns : ℕ) :
  num_colors = 4 →
  num_patterns = 5 →
  num_colors * num_patterns = 20 :=
by sorry

end dress_design_combinations_l3775_377571


namespace quadratic_inequality_range_l3775_377592

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 15/2*a > 0) ↔ a > 5/6 := by
  sorry

end quadratic_inequality_range_l3775_377592


namespace total_cost_of_bottle_caps_l3775_377566

/-- The cost of a single bottle cap in dollars -/
def bottle_cap_cost : ℕ := 2

/-- The number of bottle caps -/
def num_bottle_caps : ℕ := 6

/-- Theorem: The total cost of 6 bottle caps is $12 -/
theorem total_cost_of_bottle_caps : 
  bottle_cap_cost * num_bottle_caps = 12 := by
  sorry

end total_cost_of_bottle_caps_l3775_377566


namespace rectangular_prism_diagonals_sum_l3775_377510

theorem rectangular_prism_diagonals_sum 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 50) 
  (h2 : x*y + y*z + z*x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 := by
  sorry

end rectangular_prism_diagonals_sum_l3775_377510


namespace divisibility_condition_l3775_377547

def a (n : ℕ) : ℕ := 3 * 4^n

theorem divisibility_condition (n : ℕ) :
  (∀ m : ℕ, 1992 ∣ (m^(a n + 6) - m^(a n + 4) - m^5 + m^3)) ↔ Odd n :=
sorry

end divisibility_condition_l3775_377547


namespace area_ratio_triangle_to_hexagon_l3775_377520

/-- A regular hexagon with vertices labeled A, B, C, D, E, F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon. -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Triangle formed by connecting every second vertex of the hexagon. -/
def triangle_ACE (h : RegularHexagon) : sorry := sorry

/-- The area of triangle ACE. -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the ratio of the area of triangle ACE to the area of the regular hexagon is 2/3. -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  (area_triangle_ACE h) / (area_hexagon h) = 2/3 := by sorry

end area_ratio_triangle_to_hexagon_l3775_377520


namespace circle_equation_from_diameter_l3775_377589

/-- Given a circle with diameter endpoints (2, 0) and (2, -2), its equation is (x - 2)² + (y + 1)² = 1 -/
theorem circle_equation_from_diameter (x y : ℝ) :
  let endpoint1 : ℝ × ℝ := (2, 0)
  let endpoint2 : ℝ × ℝ := (2, -2)
  (x - 2)^2 + (y + 1)^2 = 1 := by sorry

end circle_equation_from_diameter_l3775_377589


namespace existence_of_special_number_l3775_377521

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) ∧
    (∀ p : Nat, p ∉ P → Nat.Prime p → ¬∃ a b : Nat, a > 0 ∧ b > 0 ∧ x = a^p + b^p) :=
by sorry

end existence_of_special_number_l3775_377521


namespace domain_of_f_x_squared_l3775_377516

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ := Set.Icc (-2) 3

-- Define the property that f(x+1) has domain [-2, 3]
def f_x_plus_1_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ≠ 0

-- Theorem statement
theorem domain_of_f_x_squared (f : ℝ → ℝ) 
  (h : f_x_plus_1_domain f) : 
  {x : ℝ | f (x^2) ≠ 0} = Set.Icc (-2) 2 := by sorry

end domain_of_f_x_squared_l3775_377516


namespace soil_bags_needed_l3775_377509

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 :=
by sorry

end soil_bags_needed_l3775_377509


namespace intended_profit_percentage_l3775_377596

/-- Given a cost price, labeled price, and selling price satisfying certain conditions,
    prove that the intended profit percentage is 1/3. -/
theorem intended_profit_percentage
  (C L S : ℝ)  -- Cost price, Labeled price, Selling price
  (P : ℝ)      -- Intended profit percentage (as a decimal)
  (h1 : L = C * (1 + P))        -- Labeled price condition
  (h2 : S = 0.90 * L)           -- 10% discount condition
  (h3 : S = 1.17 * C)           -- 17% actual profit condition
  : P = 1 / 3 := by
  sorry


end intended_profit_percentage_l3775_377596


namespace square_root_calculations_l3775_377504

theorem square_root_calculations :
  (3 * Real.sqrt 8 - Real.sqrt 32 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 6 * Real.sqrt 2 / Real.sqrt 3 = 2) ∧
  ((Real.sqrt 24 + Real.sqrt (1/6)) / Real.sqrt 3 = 13 * Real.sqrt 2 / 6) ∧
  (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 - 1 = 0 := by
  sorry

end square_root_calculations_l3775_377504


namespace max_value_trig_product_max_value_trig_product_achievable_l3775_377576

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
  (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) ≤ 4.5 :=
by sorry

theorem max_value_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
    (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) = 4.5 :=
by sorry

end max_value_trig_product_max_value_trig_product_achievable_l3775_377576


namespace circle_radius_problem_l3775_377565

-- Define the circle and points
def Circle : Type := ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the radius of the circle
def radius (c : Circle) : ℝ := sorry

-- Define the center of the circle
def center (c : Circle) : Point := sorry

-- Define a point on the circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a secant
def is_secant (p q r : Point) (c : Circle) : Prop := 
  ¬(on_circle p c) ∧ on_circle q c ∧ on_circle r c

-- Theorem statement
theorem circle_radius_problem (c : Circle) (p q r : Point) :
  distance p (center c) = 17 →
  is_secant p q r c →
  distance p q = 11 →
  distance q r = 8 →
  radius c = 4 * Real.sqrt 5 := by sorry

end circle_radius_problem_l3775_377565


namespace exam_mistakes_l3775_377595

theorem exam_mistakes (bryan_score jen_score sammy_score total_points : ℕ) : 
  bryan_score = 20 →
  jen_score = bryan_score + 10 →
  sammy_score = jen_score - 2 →
  total_points = 35 →
  total_points - sammy_score = 7 :=
by sorry

end exam_mistakes_l3775_377595


namespace greatest_two_digit_with_digit_product_12_l3775_377543

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end greatest_two_digit_with_digit_product_12_l3775_377543


namespace coins_value_l3775_377519

/-- Represents the total value of coins in cents -/
def total_value (total_coins : ℕ) (nickels : ℕ) : ℕ :=
  let dimes : ℕ := total_coins - nickels
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 50 total coins, 30 of which are nickels, the total value is $3.50 -/
theorem coins_value : total_value 50 30 = 350 := by
  sorry

end coins_value_l3775_377519


namespace mets_fans_count_l3775_377568

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  dodgers : ℕ
  red_sox : ℕ
  cubs : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 585

/-- Checks if the given fan counts satisfy the specified ratios -/
def satisfies_ratios (fc : FanCounts) : Prop :=
  3 * fc.mets = 2 * fc.yankees ∧
  3 * fc.dodgers = fc.mets ∧
  4 * fc.red_sox = 5 * fc.mets ∧
  2 * fc.cubs = fc.mets

/-- Checks if the sum of all fan counts equals the total number of fans -/
def sums_to_total (fc : FanCounts) : Prop :=
  fc.yankees + fc.mets + fc.dodgers + fc.red_sox + fc.cubs = total_fans

/-- The main theorem stating that there are 120 NY Mets fans -/
theorem mets_fans_count :
  ∃ (fc : FanCounts), satisfies_ratios fc ∧ sums_to_total fc ∧ fc.mets = 120 :=
sorry

end mets_fans_count_l3775_377568


namespace day_after_53_days_l3775_377540

-- Define the days of the week
inductive DayOfWeek
  | Friday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday

-- Define a function to advance the day by one
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday

-- Define a function to advance the day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem day_after_53_days : advanceDay DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end day_after_53_days_l3775_377540


namespace arrange_seven_white_five_black_l3775_377550

/-- The number of ways to arrange white and black balls with no adjacent black balls -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + black - black + 1) (black + 1)

/-- Theorem stating that arranging 7 white and 5 black balls with no adjacent black balls results in 56 ways -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

#eval arrangeBalls 7 5

end arrange_seven_white_five_black_l3775_377550


namespace money_distribution_l3775_377563

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 1000)
  (ac_sum : A + C = 700)
  (bc_sum : B + C = 600) :
  C = 300 := by
  sorry

end money_distribution_l3775_377563


namespace students_above_eight_l3775_377522

theorem students_above_eight (total : ℕ) (below_eight : ℕ) (eight : ℕ) (above_eight : ℕ) : 
  total = 80 →
  below_eight = total / 4 →
  eight = 36 →
  above_eight = 2 * eight / 3 →
  above_eight = 24 := by
sorry

end students_above_eight_l3775_377522


namespace sin_300_degrees_l3775_377530

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l3775_377530


namespace product_of_odot_l3775_377575

def A : Finset Int := {-2, 1}
def B : Finset Int := {-1, 2}

def odot (A B : Finset Int) : Finset Int :=
  (A.product B).image (fun (x : Int × Int) => x.1 * x.2)

theorem product_of_odot :
  (odot A B).prod id = 8 := by sorry

end product_of_odot_l3775_377575


namespace b_grazing_months_l3775_377559

/-- Represents the number of months B put his oxen for grazing -/
def b_months : ℕ := sorry

/-- Total rent of the pasture in Rs. -/
def total_rent : ℕ := 280

/-- C's share of the rent in Rs. -/
def c_share : ℕ := 72

/-- Calculates the total oxen-months for all farmers -/
def total_oxen_months : ℕ := 10 * 7 + 12 * b_months + 15 * 3

/-- Theorem stating that B put his oxen for grazing for 5 months -/
theorem b_grazing_months : b_months = 5 := by
  sorry

end b_grazing_months_l3775_377559


namespace systematic_sample_fourth_number_l3775_377590

def class_size : ℕ := 48
def sample_size : ℕ := 4
def interval : ℕ := class_size / sample_size

def is_valid_sample (s : Finset ℕ) : Prop :=
  s.card = sample_size ∧ 
  ∀ x ∈ s, 1 ≤ x ∧ x ≤ class_size ∧
  ∃ k : ℕ, x = 1 + k * interval

theorem systematic_sample_fourth_number :
  ∀ s : Finset ℕ, is_valid_sample s →
  (5 ∈ s ∧ 29 ∈ s ∧ 41 ∈ s) →
  17 ∈ s :=
by sorry

end systematic_sample_fourth_number_l3775_377590
