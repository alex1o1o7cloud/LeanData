import Mathlib

namespace rope_pieces_needed_l2447_244746

/-- The number of stories Tom needs to lower the rope --/
def stories : ℕ := 6

/-- The height of one story in feet --/
def story_height : ℕ := 10

/-- The length of one piece of rope in feet --/
def rope_length : ℕ := 20

/-- The percentage of rope lost when lashing pieces together --/
def rope_loss_percentage : ℚ := 1/4

/-- The number of pieces of rope Tom needs to buy --/
def pieces_needed : ℕ := 4

theorem rope_pieces_needed :
  (stories * story_height : ℚ) ≤ pieces_needed * (rope_length * (1 - rope_loss_percentage)) ∧
  (stories * story_height : ℚ) > (pieces_needed - 1) * (rope_length * (1 - rope_loss_percentage)) :=
sorry

end rope_pieces_needed_l2447_244746


namespace ratio_G_to_N_l2447_244795

-- Define the variables
variable (N : ℝ) -- Number of non-college graduates
variable (C : ℝ) -- Number of college graduates without a graduate degree
variable (G : ℝ) -- Number of college graduates with a graduate degree

-- Define the conditions
axiom ratio_C_to_N : C = (2/3) * N
axiom prob_G : G / (G + C) = 0.15789473684210525

-- Theorem to prove
theorem ratio_G_to_N : G = (1/8) * N := by sorry

end ratio_G_to_N_l2447_244795


namespace ball_hit_ground_time_l2447_244711

/-- The time when a ball hits the ground -/
theorem ball_hit_ground_time : ∃ (t : ℚ),
  t = 2313 / 1000 ∧
  0 = -4.9 * t^2 + 7 * t + 10 :=
by sorry

end ball_hit_ground_time_l2447_244711


namespace samosa_price_is_two_l2447_244705

/-- Represents the cost of a meal at Delicious Delhi restaurant --/
structure MealCost where
  samosa_price : ℝ
  samosa_quantity : ℕ
  pakora_price : ℝ
  pakora_quantity : ℕ
  lassi_price : ℝ
  tip_percentage : ℝ
  total_with_tax : ℝ

/-- Theorem stating that the samosa price is $2 given the conditions of Hilary's meal --/
theorem samosa_price_is_two (meal : MealCost) : meal.samosa_price = 2 :=
  by
  have h1 : meal.samosa_quantity = 3 := by sorry
  have h2 : meal.pakora_price = 3 := by sorry
  have h3 : meal.pakora_quantity = 4 := by sorry
  have h4 : meal.lassi_price = 2 := by sorry
  have h5 : meal.tip_percentage = 0.25 := by sorry
  have h6 : meal.total_with_tax = 25 := by sorry
  
  -- The proof would go here
  sorry

end samosa_price_is_two_l2447_244705


namespace smallest_integer_with_remainder_l2447_244752

theorem smallest_integer_with_remainder (n : ℕ) : n = 169 →
  n > 16 ∧
  n % 6 = 1 ∧
  n % 7 = 1 ∧
  n % 8 = 1 ∧
  ∀ m : ℕ, m > 16 ∧ m % 6 = 1 ∧ m % 7 = 1 ∧ m % 8 = 1 → n ≤ m :=
by sorry

end smallest_integer_with_remainder_l2447_244752


namespace intersection_when_m_3_range_of_m_when_B_subset_A_l2447_244701

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 4 ≤ x ∧ x ≤ 5} := by sorry

theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A → m ≤ 3 := by sorry

end intersection_when_m_3_range_of_m_when_B_subset_A_l2447_244701


namespace symmetric_linear_factor_implies_quadratic_factor_l2447_244758

-- Define a polynomial in two variables
variable (P : ℝ → ℝ → ℝ)

-- Define the property of being symmetric
def IsSymmetric (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, P x y = P y x

-- Define the property of having (x - y) as a factor
def HasLinearFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ Q : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y) * Q x y

-- Define the property of having (x - y)² as a factor
def HasQuadraticFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ R : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y)^2 * R x y

-- State the theorem
theorem symmetric_linear_factor_implies_quadratic_factor
  (hSymmetric : IsSymmetric P) (hLinearFactor : HasLinearFactor P) :
  HasQuadraticFactor P := by
  sorry

end symmetric_linear_factor_implies_quadratic_factor_l2447_244758


namespace volleyball_team_selection_l2447_244736

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 16 →
  k = 7 →
  m = 2 →
  (Nat.choose (n - m) k) + (Nat.choose (n - m) (k - m)) = 5434 :=
by sorry

end volleyball_team_selection_l2447_244736


namespace nine_not_in_remaining_sums_l2447_244735

/-- Represents a cube with numbered faces -/
structure NumberedCube where
  faces : Fin 6 → Nat
  face_values : ∀ i, faces i ∈ Finset.range 7
  distinct_faces : ∀ i j, i ≠ j → faces i ≠ faces j
  opposite_pair_sum_11 : ∃ i j, i ≠ j ∧ faces i + faces j = 11

/-- The sum of the remaining pairs of opposite faces -/
def remaining_pair_sums (cube : NumberedCube) : Finset Nat :=
  sorry

theorem nine_not_in_remaining_sums (cube : NumberedCube) :
  9 ∉ remaining_pair_sums cube :=
sorry

end nine_not_in_remaining_sums_l2447_244735


namespace cube_with_cylindrical_hole_volume_l2447_244741

/-- The volume of a cube with a cylindrical hole -/
theorem cube_with_cylindrical_hole_volume (cube_side : ℝ) (hole_diameter : ℝ) : 
  cube_side = 6 →
  hole_diameter = 3 →
  abs (cube_side ^ 3 - π * (hole_diameter / 2) ^ 2 * cube_side - 173.59) < 0.01 := by
  sorry

end cube_with_cylindrical_hole_volume_l2447_244741


namespace ab_value_l2447_244744

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end ab_value_l2447_244744


namespace max_distance_to_line_l2447_244776

/-- Given a line ax + by + c = 0 where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin (0, 0) to this line is √5. -/
theorem max_distance_to_line (a b c : ℝ) :
  (a + c = 2 * b) →  -- arithmetic sequence condition
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- line exists
  (∀ (x y : ℝ), a * x + b * y + c = 0 → (x^2 + y^2 : ℝ) ≤ 5) ∧
  (∃ (x y : ℝ), a * x + b * y + c = 0 ∧ x^2 + y^2 = 5) :=
by sorry

end max_distance_to_line_l2447_244776


namespace min_floor_sum_l2447_244785

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (min : ℕ), min = 3 ∧
  (⌊(a^2 + b^2) / (a + b)⌋ + ⌊(b^2 + c^2) / (b + c)⌋ + ⌊(c^2 + a^2) / (c + a)⌋ ≥ min) ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    ⌊(x^2 + y^2) / (x + y)⌋ + ⌊(y^2 + z^2) / (y + z)⌋ + ⌊(z^2 + x^2) / (z + x)⌋ ≥ min :=
by sorry

end min_floor_sum_l2447_244785


namespace complex_sum_problem_l2447_244747

theorem complex_sum_problem (a b c d g h : ℂ) : 
  b = 4 →
  g = -a - c →
  a + b * I + c + d * I + g + h * I = 3 * I →
  d + h = -1 := by sorry

end complex_sum_problem_l2447_244747


namespace price_decrease_percentage_l2447_244755

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let decrease := original_price - sale_price
  let percentage_decrease := (decrease / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε := by
  sorry

end price_decrease_percentage_l2447_244755


namespace jonathan_distance_l2447_244787

theorem jonathan_distance (J : ℝ) 
  (mercedes_distance : ℝ → ℝ)
  (davonte_distance : ℝ → ℝ)
  (h1 : mercedes_distance J = 2 * J)
  (h2 : davonte_distance J = mercedes_distance J + 2)
  (h3 : mercedes_distance J + davonte_distance J = 32) :
  J = 7.5 := by
sorry

end jonathan_distance_l2447_244787


namespace min_value_expression_l2447_244790

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 5) : x^2 + y^2 + 2*z^2 - x^2*y^2*z ≥ -6 := by
  sorry

end min_value_expression_l2447_244790


namespace complex_fraction_equality_l2447_244713

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end complex_fraction_equality_l2447_244713


namespace min_students_all_activities_l2447_244739

theorem min_students_all_activities 
  (total : Nat) 
  (swim : Nat) 
  (cycle : Nat) 
  (tennis : Nat) 
  (h1 : total = 52) 
  (h2 : swim = 30) 
  (h3 : cycle = 35) 
  (h4 : tennis = 42) :
  total - ((total - swim) + (total - cycle) + (total - tennis)) = 3 := by
  sorry

end min_students_all_activities_l2447_244739


namespace common_root_value_l2447_244791

-- Define the polynomials
def poly1 (x C : ℝ) : ℝ := x^3 + C*x^2 + 15
def poly2 (x D : ℝ) : ℝ := x^3 + D*x + 35

-- Theorem statement
theorem common_root_value (C D : ℝ) :
  ∃ (p : ℝ), 
    (poly1 p C = 0 ∧ poly2 p D = 0) ∧ 
    (∃ (q r : ℝ), p * q * r = -15) ∧
    (∃ (s t : ℝ), p * s * t = -35) →
    p = Real.rpow 525 (1/3) := by
  sorry

end common_root_value_l2447_244791


namespace second_number_is_22_l2447_244768

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end second_number_is_22_l2447_244768


namespace time_to_reach_ticket_window_l2447_244706

-- Define Kit's movement and remaining distance
def initial_distance : ℝ := 90 -- feet
def initial_time : ℝ := 30 -- minutes
def remaining_distance : ℝ := 100 -- yards

-- Define conversion factor
def yards_to_feet : ℝ := 3 -- feet per yard

-- Theorem to prove
theorem time_to_reach_ticket_window : 
  (remaining_distance * yards_to_feet) / (initial_distance / initial_time) = 100 := by
  sorry

end time_to_reach_ticket_window_l2447_244706


namespace inequality_solution_l2447_244726

theorem inequality_solution (x : ℝ) : 
  (x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5) →
  (1 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 1 / (x - 5) < 1 / 24) ↔ 
  (x < -2 ∨ (1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end inequality_solution_l2447_244726


namespace parallelogram_smaller_angle_measure_l2447_244727

/-- 
A parallelogram with one angle exceeding the other by 50 degrees has a smaller angle of 65 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  (a = b + 50) → (a + b = 180) →
  b = 65 := by
sorry

end parallelogram_smaller_angle_measure_l2447_244727


namespace fifth_term_of_geometric_sequence_l2447_244753

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem fifth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_third : a 3 = -4) 
  (h_seventh : a 7 = -16) : 
  a 5 = -8 := by
sorry

end fifth_term_of_geometric_sequence_l2447_244753


namespace average_weight_problem_l2447_244750

/-- Given the average weight of three people and two subsets of them, prove the average weight of the remaining subset. -/
theorem average_weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)  -- Average weight of a, b, and c is 60 kg
  (h2 : (b + c) / 2 = 50)      -- Average weight of b and c is 50 kg
  (h3 : b = 60)                -- Weight of b is 60 kg
  : (a + b) / 2 = 70 :=        -- Average weight of a and b is 70 kg
by sorry

end average_weight_problem_l2447_244750


namespace fixed_point_coordinates_l2447_244730

/-- Given that for any real number k, the line (3+k)x + (1-2k)y + 1 + 5k = 0
    passes through a fixed point A, prove that the coordinates of A are (-1, 2). -/
theorem fixed_point_coordinates (A : ℝ × ℝ) :
  (∀ k : ℝ, (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0) →
  A = (-1, 2) := by
  sorry

end fixed_point_coordinates_l2447_244730


namespace parallel_lines_a_value_l2447_244771

/-- Two lines in the plane, represented by their equations --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel --/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are identical --/
def identical (l₁ l₂ : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

theorem parallel_lines_a_value (a : ℝ) :
  let l₁ : Line := { a := a, b := 3, c := 1 }
  let l₂ : Line := { a := 2, b := a + 1, c := 1 }
  parallel l₁ l₂ ∧ ¬ identical l₁ l₂ → a = -3 := by
  sorry

end parallel_lines_a_value_l2447_244771


namespace larry_stickers_l2447_244762

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends up with -/
def final_stickers : ℕ := 87

/-- The initial number of stickers Larry had -/
def initial_stickers : ℕ := final_stickers + lost_stickers

theorem larry_stickers : initial_stickers = 93 := by
  sorry

end larry_stickers_l2447_244762


namespace zeros_in_20_pow_10_eq_11_l2447_244712

/-- The number of zeros in the decimal representation of 20^10 -/
def zeros_in_20_pow_10 : ℕ :=
  let base_20_pow_10 := (20 : ℕ) ^ 10
  let digits := base_20_pow_10.digits 10
  digits.count 0

/-- Theorem stating that the number of zeros in 20^10 is 11 -/
theorem zeros_in_20_pow_10_eq_11 : zeros_in_20_pow_10 = 11 := by
  sorry

end zeros_in_20_pow_10_eq_11_l2447_244712


namespace square_number_ratio_l2447_244732

theorem square_number_ratio (k : ℕ) (h : k ≥ 2) :
  ∀ a b : ℕ, a ≠ 0 → b ≠ 0 →
  (a^2 + b^2) / (a * b + 1) = k^2 ↔ a = k ∧ b = k^3 := by
sorry

end square_number_ratio_l2447_244732


namespace negation_equivalence_l2447_244797

theorem negation_equivalence :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end negation_equivalence_l2447_244797


namespace problem_solution_l2447_244734

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a * Real.sqrt a + b * Real.sqrt b = 183)
  (eq2 : a * Real.sqrt b + b * Real.sqrt a = 182) :
  9 / 5 * (a + b) = 657 := by
  sorry

end problem_solution_l2447_244734


namespace equation_solutions_l2447_244751

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define our equation
def equation (x : ℝ) : Prop := (floor x : ℝ) * (x^2 + 1) = x^3

-- Theorem statement
theorem equation_solutions :
  (∀ k : ℕ, ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ equation x) ∧
  (∀ x : ℝ, x > 0 → equation x → ¬ (∃ q : ℚ, (q : ℝ) = x)) :=
sorry

end equation_solutions_l2447_244751


namespace longest_side_of_triangle_l2447_244738

theorem longest_side_of_triangle (x : ℝ) : 
  9 + (x + 5) + (2*x + 3) = 40 →
  max 9 (max (x + 5) (2*x + 3)) = 55/3 :=
by sorry

end longest_side_of_triangle_l2447_244738


namespace quarters_percentage_theorem_l2447_244716

def num_dimes : ℕ := 70
def num_quarters : ℕ := 30
def num_nickels : ℕ := 15

def value_dime : ℕ := 10
def value_quarter : ℕ := 25
def value_nickel : ℕ := 5

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
def quarters_value : ℕ := num_quarters * value_quarter

theorem quarters_percentage_theorem : 
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 750 / 1525 * 100 := by
  sorry

end quarters_percentage_theorem_l2447_244716


namespace count_perfect_square_factors_360_factorization_360_l2447_244704

/-- The number of perfect square factors of 360 -/
def perfect_square_factors_360 : ℕ :=
  4

theorem count_perfect_square_factors_360 :
  perfect_square_factors_360 = 4 := by
  sorry

/-- Prime factorization of 360 -/
theorem factorization_360 : 360 = 2^3 * 3^2 * 5 := by
  sorry

end count_perfect_square_factors_360_factorization_360_l2447_244704


namespace prime_square_product_equality_l2447_244782

theorem prime_square_product_equality (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2)
  (h_x_range : x ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_y_range : y ∈ Finset.range ((p - 1) / 2 + 1) \ {0})
  (h_square : ∃ k : ℕ, x * (p - x) * y * (p - y) = k^2) :
  x = y := by
sorry

end prime_square_product_equality_l2447_244782


namespace counterclockwise_notation_l2447_244759

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with its direction and angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Converts a rotation to its signed angle representation -/
def Rotation.toSignedAngle (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation (angle : ℝ) :
  (Rotation.toSignedAngle { direction := RotationDirection.Counterclockwise, angle := angle }) = -angle :=
by sorry

end counterclockwise_notation_l2447_244759


namespace f_monotone_increasing_iff_a_in_range_l2447_244742

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * sin (2*x) + a * sin x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 - (2/3) * cos (2*x) + a * cos x

/-- Theorem stating the range of 'a' for which f(x) is monotonically increasing -/
theorem f_monotone_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-1/3) (1/3) :=
by sorry

end f_monotone_increasing_iff_a_in_range_l2447_244742


namespace f_at_2_l2447_244769

theorem f_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 5 * x - 1) : f 2 = 21 := by
  sorry

end f_at_2_l2447_244769


namespace chad_odd_jobs_income_l2447_244702

theorem chad_odd_jobs_income 
  (savings_rate : Real)
  (mowing_income : Real)
  (birthday_income : Real)
  (videogame_income : Real)
  (total_savings : Real)
  (h1 : savings_rate = 0.4)
  (h2 : mowing_income = 600)
  (h3 : birthday_income = 250)
  (h4 : videogame_income = 150)
  (h5 : total_savings = 460) :
  ∃ (odd_jobs_income : Real),
    odd_jobs_income = 150 ∧
    total_savings = savings_rate * (mowing_income + birthday_income + videogame_income + odd_jobs_income) :=
by
  sorry


end chad_odd_jobs_income_l2447_244702


namespace odd_prime_condition_l2447_244763

theorem odd_prime_condition (p : ℕ) (h_prime : Nat.Prime p) : 
  (∃! k : ℕ, Even k ∧ k ∣ (14 * p)) → Odd p :=
sorry

end odd_prime_condition_l2447_244763


namespace first_discount_rate_l2447_244796

/-- Proves that given a shirt with an original price of 400, which after two
    consecutive discounts (the second being 5%) results in a final price of 340,
    the first discount rate is equal to (200/19)%. -/
theorem first_discount_rate (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 340 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 200 / 19 / 100 := by
  sorry

end first_discount_rate_l2447_244796


namespace twenty_paise_coins_l2447_244728

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 500 ∧ 
  total_value = 105 ∧ 
  ∃ (x y : ℕ), x + y = total_coins ∧ 
                (20 : ℚ)/100 * x + (25 : ℚ)/100 * y = total_value →
  x = 400 :=
by sorry

end twenty_paise_coins_l2447_244728


namespace potato_wedges_count_l2447_244783

/-- The number of wedges one potato can be cut into -/
def wedges_per_potato : ℕ := sorry

/-- The total number of potatoes harvested -/
def total_potatoes : ℕ := 67

/-- The number of potatoes cut into wedges -/
def wedge_potatoes : ℕ := 13

/-- The number of potato chips one potato can make -/
def chips_per_potato : ℕ := 20

/-- The difference between the number of potato chips and wedges -/
def chip_wedge_difference : ℕ := 436

theorem potato_wedges_count :
  wedges_per_potato = 8 ∧
  (total_potatoes - wedge_potatoes) / 2 * chips_per_potato - wedge_potatoes * wedges_per_potato = chip_wedge_difference :=
by sorry

end potato_wedges_count_l2447_244783


namespace chocolate_theorem_l2447_244740

def chocolate_problem (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box

theorem chocolate_theorem : chocolate_problem 12 6 30 = 7 := by
  sorry

end chocolate_theorem_l2447_244740


namespace seating_arrangements_l2447_244729

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people : ℕ := 3

/-- The number of seats that must be left empty at the ends -/
def end_seats : ℕ := 2

/-- The number of seats available for seating after accounting for end seats -/
def available_seats : ℕ := total_seats - end_seats

/-- The number of gaps between seated people (including before first and after last) -/
def gaps : ℕ := people + 1

/-- Theorem stating the number of seating arrangements -/
theorem seating_arrangements :
  (Nat.choose available_seats gaps) * (Nat.factorial people) = 24 := by
  sorry

end seating_arrangements_l2447_244729


namespace red_marble_probability_l2447_244780

/-- The probability of drawing exactly k red marbles out of n draws with replacement
    from a bag containing r red marbles and b blue marbles. -/
def probability (r b k n : ℕ) : ℚ :=
  (n.choose k) * ((r : ℚ) / (r + b : ℚ)) ^ k * ((b : ℚ) / (r + b : ℚ)) ^ (n - k)

/-- The probability of drawing exactly 4 red marbles out of 8 draws with replacement
    from a bag containing 8 red marbles and 4 blue marbles is equal to 1120/6561. -/
theorem red_marble_probability : probability 8 4 4 8 = 1120 / 6561 := by
  sorry

end red_marble_probability_l2447_244780


namespace circle_max_area_l2447_244714

/-- Given a circle equation with parameter m, prove that when the area is maximum, 
    the standard equation of the circle is (x-1)^2 + (y+3)^2 = 1 -/
theorem circle_max_area (x y m : ℝ) : 
  (∃ r, x^2 + y^2 - 2*x + 2*m*y + 2*m^2 - 6*m + 9 = 0 ↔ (x-1)^2 + (y+m)^2 = r^2) →
  (∀ m', ∃ r', x^2 + y^2 - 2*x + 2*m'*y + 2*m'^2 - 6*m' + 9 = 0 → 
    (x-1)^2 + (y+m')^2 = r'^2 ∧ r'^2 ≤ 1) →
  (x-1)^2 + (y+3)^2 = 1 := by sorry

end circle_max_area_l2447_244714


namespace quadratic_nature_l2447_244754

/-- Given a quadratic function f(x) = ax^2 + bx + b^2 / (3a), prove that:
    1. If a > 0, the graph of y = f(x) has a minimum
    2. If a < 0, the graph of y = f(x) has a maximum -/
theorem quadratic_nature (a b : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + b^2 / (3 * a)
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) :=
sorry

end quadratic_nature_l2447_244754


namespace circle_center_center_coordinates_l2447_244756

theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ↔ 
  ((x - 1)^2 + (y + 2)^2 = 0) :=
sorry

theorem center_coordinates : 
  ∃ (x y : ℝ), (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ∧ 
  (x = 1 ∧ y = -2) :=
sorry

end circle_center_center_coordinates_l2447_244756


namespace congruent_rectangle_perimeter_l2447_244793

/-- Given a rectangle with sides y and z, and a square with side x placed against
    the shorter side y, the perimeter of one of the four congruent rectangles
    formed in the remaining space is equal to 2y + 2z - 4x. -/
theorem congruent_rectangle_perimeter 
  (y z x : ℝ) 
  (h1 : y > 0) 
  (h2 : z > 0) 
  (h3 : x > 0) 
  (h4 : x < y) 
  (h5 : x < z) : 
  2*y + 2*z - 4*x = 2*((y - x) + (z - x)) := by
  sorry


end congruent_rectangle_perimeter_l2447_244793


namespace solution_satisfies_equation_all_solutions_are_general_l2447_244720

/-- The differential equation -/
def diff_eq (x y : ℝ) : Prop :=
  ∃ (dx dy : ℝ), (y^3 - 2*x*y) * dx + (3*x*y^2 - x^2) * dy = 0

/-- The general solution -/
def general_solution (x y C : ℝ) : Prop :=
  y^3 * x - x^2 * y = C

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem solution_satisfies_equation :
  ∀ (x y C : ℝ), general_solution x y C → diff_eq x y :=
by sorry

/-- Theorem stating that any solution to the differential equation is of the form of the general solution -/
theorem all_solutions_are_general :
  ∀ (x y : ℝ), diff_eq x y → ∃ (C : ℝ), general_solution x y C :=
by sorry

end solution_satisfies_equation_all_solutions_are_general_l2447_244720


namespace sureshs_speed_l2447_244778

/-- Suresh's walking speed problem -/
theorem sureshs_speed (track_circumference : ℝ) (meeting_time : ℝ) (wife_speed : ℝ) 
  (h1 : track_circumference = 726) 
  (h2 : meeting_time = 5.28)
  (h3 : wife_speed = 3.75) : 
  ∃ (suresh_speed : ℝ), abs (suresh_speed - 4.5054) < 0.0001 := by
  sorry

end sureshs_speed_l2447_244778


namespace total_cups_doubled_is_60_l2447_244786

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients for a doubled recipe -/
def totalCupsDoubled (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  let butterCups := ratio.butter * partSize
  let sugarCups := ratio.sugar * partSize
  2 * (butterCups + flourCups + sugarCups)

/-- Theorem: Given the recipe ratio and flour quantity, the total cups for a doubled recipe is 60 -/
theorem total_cups_doubled_is_60 :
  totalCupsDoubled ⟨2, 5, 3⟩ 15 = 60 := by
  sorry

end total_cups_doubled_is_60_l2447_244786


namespace timothy_chickens_l2447_244710

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cows_cost : ℕ := 20 * 1000
def solar_panel_cost : ℕ := 6 * 100 + 6000
def chicken_price : ℕ := 5
def total_cost : ℕ := 147700

theorem timothy_chickens :
  ∃ (num_chickens : ℕ),
    land_cost + house_cost + cows_cost + solar_panel_cost + num_chickens * chicken_price = total_cost ∧
    num_chickens = 100 :=
by sorry

end timothy_chickens_l2447_244710


namespace certain_number_value_l2447_244760

theorem certain_number_value (x y z : ℝ) 
  (h1 : 0.5 * x = y + z) 
  (h2 : x - 2 * y = 40) : 
  z = 20 := by sorry

end certain_number_value_l2447_244760


namespace measure_six_pints_l2447_244765

/-- Represents the state of wine distribution -/
structure WineState :=
  (total : ℕ)
  (container8 : ℕ)
  (container5 : ℕ)

/-- Represents a pouring action -/
inductive PourAction
  | FillFrom8To5
  | FillFrom5To8
  | EmptyTo8
  | EmptyTo5
  | Empty8
  | Empty5

/-- Applies a pouring action to a wine state -/
def applyAction (state : WineState) (action : PourAction) : WineState :=
  match action with
  | PourAction.FillFrom8To5 => sorry
  | PourAction.FillFrom5To8 => sorry
  | PourAction.EmptyTo8 => sorry
  | PourAction.EmptyTo5 => sorry
  | PourAction.Empty8 => sorry
  | PourAction.Empty5 => sorry

/-- Checks if the goal state is reached -/
def isGoalState (state : WineState) : Prop :=
  state.container8 = 6

/-- Theorem: It is possible to measure 6 pints into the 8-pint container -/
theorem measure_six_pints 
  (initialState : WineState)
  (h_total : initialState.total = 12)
  (h_containers : initialState.container8 = 0 ∧ initialState.container5 = 0) :
  ∃ (actions : List PourAction), 
    isGoalState (actions.foldl applyAction initialState) :=
sorry

end measure_six_pints_l2447_244765


namespace three_from_fifteen_combination_l2447_244731

theorem three_from_fifteen_combination : (Nat.choose 15 3) = 455 := by sorry

end three_from_fifteen_combination_l2447_244731


namespace vector_combination_vectors_parallel_l2447_244745

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The theorem states that a = (5/9)b + (8/9)c -/
theorem vector_combination : a = (5/9 • b) + (8/9 • c) := by sorry

/-- Helper function to check if two vectors are parallel -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

/-- The theorem states that (a + kc) is parallel to (2b - a) when k = -16/13 -/
theorem vectors_parallel : are_parallel (a + (-16/13 • c)) (2 • b - a) := by sorry

end vector_combination_vectors_parallel_l2447_244745


namespace units_digit_of_p_l2447_244721

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 4) = 0 →
  units_digit p = 6 :=
by sorry

end units_digit_of_p_l2447_244721


namespace product_mod_23_l2447_244775

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 := by sorry

end product_mod_23_l2447_244775


namespace oil_depth_in_specific_tank_l2447_244743

/-- Represents a horizontally positioned cylindrical tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a cylindrical tank given the surface area -/
def oilDepth (tank : CylindricalTank) (surfaceArea : ℝ) : ℝ :=
  sorry

theorem oil_depth_in_specific_tank :
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surfaceArea : ℝ := 32
  oilDepth tank surfaceArea = 4 := by
  sorry

end oil_depth_in_specific_tank_l2447_244743


namespace sheep_buying_problem_sheep_buying_problem_unique_l2447_244792

/-- The number of people buying the sheep -/
def num_people : ℕ := 21

/-- The price of the sheep in coins -/
def sheep_price : ℕ := 150

/-- Theorem stating the solution to the sheep-buying problem -/
theorem sheep_buying_problem :
  (∃ (n : ℕ) (p : ℕ),
    n = num_people ∧
    p = sheep_price ∧
    5 * n + 45 = p ∧
    7 * n + 3 = p) :=
by sorry

/-- Theorem proving the uniqueness of the solution -/
theorem sheep_buying_problem_unique :
  ∀ (n : ℕ) (p : ℕ),
    5 * n + 45 = p ∧
    7 * n + 3 = p →
    n = num_people ∧
    p = sheep_price :=
by sorry

end sheep_buying_problem_sheep_buying_problem_unique_l2447_244792


namespace arithmetic_computation_l2447_244718

theorem arithmetic_computation : -7 * 5 - (-4 * 3) + (-9 * -6) = 31 := by
  sorry

end arithmetic_computation_l2447_244718


namespace binomial_320_320_l2447_244799

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by
  sorry

end binomial_320_320_l2447_244799


namespace triangle_base_increase_l2447_244748

theorem triangle_base_increase (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.95 * h
  let new_area := 1.045 * (0.5 * b * h)
  ∃ x : ℝ, x > 0 ∧ new_area = 0.5 * (b * (1 + x / 100)) * new_height ∧ x = 10 :=
by sorry

end triangle_base_increase_l2447_244748


namespace root_in_interval_l2447_244766

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 2.5, f x = 0) :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  sorry

end root_in_interval_l2447_244766


namespace white_line_length_l2447_244722

theorem white_line_length 
  (blue_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : blue_length = 3.33) 
  (h2 : length_difference = 4.33) : 
  blue_length + length_difference = 7.66 := by
sorry

end white_line_length_l2447_244722


namespace sample_size_example_l2447_244774

/-- Represents the sample size of a survey --/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem stating that for a population of 300 students with 50 selected, the sample size is 50 --/
theorem sample_size_example : sample_size 300 50 = 50 := by
  sorry

end sample_size_example_l2447_244774


namespace zoo_sea_lions_l2447_244777

theorem zoo_sea_lions (sea_lions : ℕ) (penguins : ℕ) : 
  (sea_lions : ℚ) / penguins = 4 / 11 →
  penguins = sea_lions + 84 →
  sea_lions = 48 := by
sorry

end zoo_sea_lions_l2447_244777


namespace scallops_per_person_is_two_l2447_244788

-- Define the constants from the problem
def scallops_per_pound : ℕ := 8
def cost_per_pound : ℚ := 24
def number_of_people : ℕ := 8
def total_cost : ℚ := 48

-- Define the function to calculate scallops per person
def scallops_per_person : ℚ :=
  (total_cost / cost_per_pound * scallops_per_pound) / number_of_people

-- Theorem to prove
theorem scallops_per_person_is_two : scallops_per_person = 2 := by
  sorry

end scallops_per_person_is_two_l2447_244788


namespace yogurt_combinations_l2447_244781

def yogurt_types : ℕ := 2
def yogurt_flavors : ℕ := 5
def topping_count : ℕ := 8

def combination_count : ℕ := yogurt_types * yogurt_flavors * (topping_count.choose 2)

theorem yogurt_combinations :
  combination_count = 280 :=
by sorry

end yogurt_combinations_l2447_244781


namespace line_circle_intersection_l2447_244717

/-- Given a circle with radius 5 and a line at distance k from its center,
    if the equation x^2 - kx + 1 = 0 has equal roots,
    then the line intersects the circle. -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - k*y + 1 = 0 → y = x)) →
  k < 5 :=
by sorry

end line_circle_intersection_l2447_244717


namespace probability_of_two_pairs_l2447_244724

def number_of_dice : ℕ := 7
def sides_per_die : ℕ := 6

def total_outcomes : ℕ := sides_per_die ^ number_of_dice

def ways_to_choose_pair_numbers : ℕ := Nat.choose 6 2
def ways_to_choose_dice_for_pairs : ℕ := Nat.choose number_of_dice 4
def ways_to_arrange_pairs : ℕ := 6  -- 4! / (2! * 2!)
def ways_to_choose_remaining_numbers : ℕ := 4 * 3 * 2

def successful_outcomes : ℕ := 
  ways_to_choose_pair_numbers * ways_to_choose_dice_for_pairs * 
  ways_to_arrange_pairs * ways_to_choose_remaining_numbers

theorem probability_of_two_pairs (h : successful_outcomes = 151200 ∧ total_outcomes = 279936) :
  (successful_outcomes : ℚ) / total_outcomes = 175 / 324 := by
  sorry

end probability_of_two_pairs_l2447_244724


namespace investment_dividend_calculation_l2447_244772

/-- Calculates the dividend received from an investment in shares with premium and dividend rates -/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem stating that the given investment yields the correct dividend -/
theorem investment_dividend_calculation :
  calculate_dividend 14400 100 (20/100) (7/100) = 840 := by
  sorry

#eval calculate_dividend 14400 100 (20/100) (7/100)

end investment_dividend_calculation_l2447_244772


namespace interest_rate_is_five_percent_l2447_244725

-- Define the principal amount and interest rate
variable (P : ℝ) -- Principal amount
variable (r : ℝ) -- Interest rate (as a decimal)

-- Define the conditions
def condition1 : Prop := P * (1 + 3 * r) = 460
def condition2 : Prop := P * (1 + 8 * r) = 560

-- Theorem to prove
theorem interest_rate_is_five_percent 
  (h1 : condition1 P r) 
  (h2 : condition2 P r) : 
  r = 0.05 := by
  sorry


end interest_rate_is_five_percent_l2447_244725


namespace mark_speeding_ticket_cost_l2447_244703

/-- Calculate the total cost of Mark's speeding ticket --/
def speeding_ticket_cost (base_fine : ℕ) (fine_increase_per_mph : ℕ) 
  (mark_speed : ℕ) (speed_limit : ℕ) (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) (lawyer_hours : ℕ) : ℕ := 
  let speed_difference := mark_speed - speed_limit
  let speed_fine := base_fine + fine_increase_per_mph * speed_difference
  let doubled_fine := 2 * speed_fine
  let total_without_lawyer := doubled_fine + court_costs
  let lawyer_cost := lawyer_fee_per_hour * lawyer_hours
  total_without_lawyer + lawyer_cost

theorem mark_speeding_ticket_cost : 
  speeding_ticket_cost 50 2 75 30 300 80 3 = 820 := by
  sorry

end mark_speeding_ticket_cost_l2447_244703


namespace car_dealership_problem_l2447_244798

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_shipment : ℕ) (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment = 80)
  (h4 : final_silver_percent = 3/10) :
  (1 - (final_silver_percent * (initial_cars + new_shipment) - initial_silver_percent * initial_cars) / new_shipment) = 13/20 := by
  sorry

end car_dealership_problem_l2447_244798


namespace no_solution_exists_l2447_244749

theorem no_solution_exists (x y : ℕ+) : 3 * y^2 ≠ x^4 + x := by
  sorry

end no_solution_exists_l2447_244749


namespace unique_self_opposite_l2447_244789

theorem unique_self_opposite : ∃! x : ℝ, x = -x := by sorry

end unique_self_opposite_l2447_244789


namespace pistachio_count_l2447_244764

theorem pistachio_count (total : ℝ) 
  (h1 : 0.95 * total * 0.75 = 57) : total = 80 := by
  sorry

end pistachio_count_l2447_244764


namespace min_value_implies_a_inequality_implies_a_range_function_properties_l2447_244737

-- Define the function f(x)
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Minimum value of f(x) is 2
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5 := by sorry

-- Part 2: Inequality holds for x ∈ [0, 1]
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2 := by sorry

-- Combined theorem
theorem function_properties (a : ℝ) :
  ((∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5) ∧
  ((∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2) := by sorry

end min_value_implies_a_inequality_implies_a_range_function_properties_l2447_244737


namespace turtle_path_max_entries_l2447_244719

/-- Represents a turtle's path on a square grid -/
structure TurtlePath (n : ℕ) :=
  (grid_size : ℕ := 4*n + 2)
  (start_corner : Bool)
  (visits_all_squares_once : Bool)
  (ends_at_start : Bool)

/-- Represents the number of times a turtle enters a row or column -/
def max_entries (path : TurtlePath n) : ℕ := sorry

/-- Main theorem: There exists a row or column that the turtle enters at least 2n + 2 times -/
theorem turtle_path_max_entries {n : ℕ} (path : TurtlePath n) 
  (h1 : path.start_corner = true)
  (h2 : path.visits_all_squares_once = true)
  (h3 : path.ends_at_start = true) :
  max_entries path ≥ 2*n + 2 :=
sorry

end turtle_path_max_entries_l2447_244719


namespace distance_AP_equals_one_l2447_244779

-- Define the triangle and circle
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (2, 0)
def center : ℝ × ℝ := (1, 1)

-- Define the inscribed circle
def ω (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define point M where ω touches BC
def M : ℝ × ℝ := (0, 0)

-- Define point P where AM intersects ω
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem distance_AP_equals_one :
  let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  d = 1 := by sorry

end distance_AP_equals_one_l2447_244779


namespace apples_left_is_340_l2447_244784

/-- The number of baskets --/
def num_baskets : ℕ := 11

/-- The number of children --/
def num_children : ℕ := 10

/-- The total number of apples initially --/
def total_apples : ℕ := 1000

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of apples picked by all children --/
def apples_picked : ℕ := num_children * sum_first_n num_baskets

/-- The number of apples left after picking --/
def apples_left : ℕ := total_apples - apples_picked

theorem apples_left_is_340 : apples_left = 340 := by
  sorry

end apples_left_is_340_l2447_244784


namespace geometric_solid_surface_area_l2447_244700

/-- Given a geometric solid that is a quarter of a cylinder with height 2,
    base area π, and radius 2, prove that its surface area is 8 + 4π. -/
theorem geometric_solid_surface_area
  (h : ℝ) (base_area : ℝ) (radius : ℝ) :
  h = 2 →
  base_area = π →
  radius = 2 →
  (2 * base_area + 2 * radius * h + (1/4) * 2 * π * radius * h) = 8 + 4 * π :=
by sorry

end geometric_solid_surface_area_l2447_244700


namespace process_time_per_picture_l2447_244715

/-- Given a total number of pictures and total processing time in hours,
    calculate the time required to process each picture in minutes. -/
def time_per_picture (total_pictures : ℕ) (total_hours : ℕ) : ℚ :=
  (total_hours * 60) / total_pictures

/-- Theorem: Given 960 pictures and a total processing time of 32 hours,
    the time required to process each picture is 2 minutes. -/
theorem process_time_per_picture :
  time_per_picture 960 32 = 2 := by
  sorry

end process_time_per_picture_l2447_244715


namespace mean_temperature_l2447_244794

theorem mean_temperature (temperatures : List ℤ) : 
  temperatures = [-10, -4, -6, -3, 0, 2, 5, 0] →
  (temperatures.sum / temperatures.length : ℚ) = -2 := by
  sorry

end mean_temperature_l2447_244794


namespace right_triangle_hypotenuse_l2447_244723

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 :=
by sorry

end right_triangle_hypotenuse_l2447_244723


namespace mans_age_twice_sons_l2447_244733

/-- Given a man who is 24 years older than his son, and the son's present age is 22,
    prove that it will take 2 years for the man's age to be twice the age of his son. -/
theorem mans_age_twice_sons (man_age son_age future_years : ℕ) : 
  man_age = son_age + 24 →
  son_age = 22 →
  future_years = 2 →
  (man_age + future_years) = 2 * (son_age + future_years) :=
by sorry

end mans_age_twice_sons_l2447_244733


namespace square_garden_perimeter_l2447_244707

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 200 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
sorry

end square_garden_perimeter_l2447_244707


namespace probability_red_or_green_l2447_244709

/-- The probability of drawing a red or green marble from a bag with specified marble counts. -/
theorem probability_red_or_green (red green blue yellow : ℕ) : 
  let total := red + green + blue + yellow
  (red + green : ℚ) / total = 9 / 14 :=
by
  sorry

#check probability_red_or_green 5 4 2 3

end probability_red_or_green_l2447_244709


namespace range_of_a_theorem_l2447_244761

def line (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 1

def opposite_sides (a : ℝ) : Prop :=
  line a 1 1 > 0 ∧ line a 0 (-2) < 0

def range_of_a : Set ℝ := { a | a < -2 ∨ a > 1/2 }

theorem range_of_a_theorem :
  ∀ a : ℝ, opposite_sides a ↔ a ∈ range_of_a := by sorry

end range_of_a_theorem_l2447_244761


namespace max_points_for_top_teams_is_76_l2447_244773

/-- Represents a soccer league with given parameters -/
structure SoccerLeague where
  numTeams : Nat
  gamesAgainstEachTeam : Nat
  pointsForWin : Nat
  pointsForDraw : Nat
  pointsForLoss : Nat

/-- Calculates the maximum possible points for each of the top three teams in the league -/
def maxPointsForTopTeams (league : SoccerLeague) : Nat :=
  sorry

/-- Theorem stating the maximum points for top teams in the specific league configuration -/
theorem max_points_for_top_teams_is_76 :
  let league : SoccerLeague := {
    numTeams := 9
    gamesAgainstEachTeam := 4
    pointsForWin := 3
    pointsForDraw := 1
    pointsForLoss := 0
  }
  maxPointsForTopTeams league = 76 := by sorry

end max_points_for_top_teams_is_76_l2447_244773


namespace modified_cube_edges_l2447_244770

/-- Represents a modified cube -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeRemoved1 : ℕ
  smallCubeRemoved2 : ℕ
  largeCubeRemoved : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (c : ModifiedCube) : ℕ := sorry

/-- Theorem stating that a specific modified cube has 22 edges -/
theorem modified_cube_edges :
  let c : ModifiedCube := {
    sideLength := 4,
    smallCubeRemoved1 := 1,
    smallCubeRemoved2 := 1,
    largeCubeRemoved := 2
  }
  edgeCount c = 22 := by sorry

end modified_cube_edges_l2447_244770


namespace divisibility_32xy76_l2447_244757

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_32xy76 (x y : ℕ) : ℕ := 320000 + 10000 * x + 1000 * y + 76

theorem divisibility_32xy76 (x y : ℕ) (hx : is_digit x) (hy : is_digit y) :
  ∃ k : ℕ, number_32xy76 x y = 4 * k :=
sorry

end divisibility_32xy76_l2447_244757


namespace count_three_digit_multiples_of_15_l2447_244708

theorem count_three_digit_multiples_of_15 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range 900)).card = 60 := by
  sorry

end count_three_digit_multiples_of_15_l2447_244708


namespace min_toothpicks_to_remove_for_given_figure_l2447_244767

/-- Represents a figure made of toothpicks -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  triangles : ℕ
  squares : ℕ

/-- The minimum number of toothpicks to remove to eliminate all shapes -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_for_given_figure :
  ∃ (figure : ToothpickFigure),
    figure.total_toothpicks = 40 ∧
    figure.triangles > 20 ∧
    figure.squares = 10 ∧
    min_toothpicks_to_remove figure = 20 := by
  sorry

end min_toothpicks_to_remove_for_given_figure_l2447_244767
