import Mathlib

namespace NUMINAMATH_CALUDE_tickets_per_friend_is_four_l3528_352890

/-- The number of tickets each friend bought on the first day -/
def tickets_per_friend : ℕ := sorry

/-- The total number of tickets to be sold -/
def total_tickets : ℕ := 80

/-- The number of friends who bought tickets on the first day -/
def num_friends : ℕ := 5

/-- The number of tickets sold on the second day -/
def second_day_tickets : ℕ := 32

/-- The number of tickets that need to be sold on the third day -/
def third_day_tickets : ℕ := 28

/-- Theorem stating that the number of tickets each friend bought on the first day is 4 -/
theorem tickets_per_friend_is_four :
  tickets_per_friend = 4 ∧
  tickets_per_friend * num_friends + second_day_tickets + third_day_tickets = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_tickets_per_friend_is_four_l3528_352890


namespace NUMINAMATH_CALUDE_total_fish_l3528_352885

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l3528_352885


namespace NUMINAMATH_CALUDE_symmetric_points_range_l3528_352821

theorem symmetric_points_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, a - x^2 = -(2*x + 1)) → a ∈ Set.Icc (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l3528_352821


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l3528_352828

theorem fourth_root_equation_solution : 
  ∃ (p q r : ℕ+), 
    4 * (7^(1/4) - 6^(1/4))^(1/4) = p^(1/4) + q^(1/4) - r^(1/4) ∧ 
    p + q + r = 99 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l3528_352828


namespace NUMINAMATH_CALUDE_parallelograms_in_triangle_l3528_352841

/-- The number of parallelograms formed inside a triangle -/
def num_parallelograms (n : ℕ) : ℕ := 3 * Nat.choose (n + 2) 4

/-- 
Theorem: The number of parallelograms formed inside a triangle 
whose sides are divided into n equal parts with parallel lines 
drawn through these points is equal to 3 * (n+2 choose 4).
-/
theorem parallelograms_in_triangle (n : ℕ) : 
  num_parallelograms n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_in_triangle_l3528_352841


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3528_352807

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| > 1} = {x : ℝ | x < 2 ∨ x > 3} := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3528_352807


namespace NUMINAMATH_CALUDE_borrowed_amount_l3528_352856

theorem borrowed_amount (P : ℝ) (interest_rate : ℝ) (total_repayment : ℝ) : 
  interest_rate = 0.1 →
  total_repayment = 1320 →
  total_repayment = P * (1 + interest_rate) →
  P = 1200 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l3528_352856


namespace NUMINAMATH_CALUDE_solve_equation_l3528_352881

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * (b ^ (1/2))

-- Theorem statement
theorem solve_equation (x : ℝ) (h : at_op 4 x = 12) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3528_352881


namespace NUMINAMATH_CALUDE_duty_arrangements_count_l3528_352806

/- Define the number of days -/
def num_days : ℕ := 7

/- Define the number of people -/
def num_people : ℕ := 4

/- Define the possible work days for each person -/
def work_days : Set ℕ := {1, 2}

/- Define the function to calculate the number of duty arrangements -/
def duty_arrangements (days : ℕ) (people : ℕ) (work_options : Set ℕ) : ℕ :=
  sorry  -- The actual calculation would go here

/- Theorem stating that the number of duty arrangements is 2520 -/
theorem duty_arrangements_count :
  duty_arrangements num_days num_people work_days = 2520 :=
sorry

end NUMINAMATH_CALUDE_duty_arrangements_count_l3528_352806


namespace NUMINAMATH_CALUDE_function_not_satisfying_differential_equation_l3528_352826

open Real

theorem function_not_satisfying_differential_equation :
  ¬∃ y : ℝ → ℝ, ∀ x : ℝ,
    (y x = (x + 1) * (Real.exp (x^2))) ∧
    (deriv y x - 2 * x * y x = 2 * x * (Real.exp (x^2))) :=
sorry

end NUMINAMATH_CALUDE_function_not_satisfying_differential_equation_l3528_352826


namespace NUMINAMATH_CALUDE_zero_in_interval_l3528_352809

-- Define the function f(x) = 2x + 3x
def f (x : ℝ) : ℝ := 2*x + 3*x

-- Theorem stating that the zero of f(x) is in the interval (-1, 0)
theorem zero_in_interval :
  ∃ x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3528_352809


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l3528_352802

-- Define a square
structure Square where
  side_length : ℝ
  angle_measure : ℝ
  is_rectangle : Prop
  similar_to_all_squares : Prop

-- Define properties of squares
axiom square_angles (s : Square) : s.angle_measure = 90

axiom square_sides_equal (s : Square) : s.side_length > 0

axiom square_is_rectangle (s : Square) : s.is_rectangle = true

axiom squares_similar (s1 s2 : Square) : s1.similar_to_all_squares ∧ s2.similar_to_all_squares

-- Theorem to prove
theorem not_all_squares_congruent : ¬∀ (s1 s2 : Square), s1 = s2 := by
  sorry


end NUMINAMATH_CALUDE_not_all_squares_congruent_l3528_352802


namespace NUMINAMATH_CALUDE_product_divisible_by_4_probability_l3528_352871

def is_divisible_by_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def count_pairs_divisible_by_4 (n : ℕ) : ℕ :=
  let even_count := n / 2
  let multiple_of_4_count := n / 4
  (even_count.choose 2) + multiple_of_4_count * (even_count - multiple_of_4_count)

theorem product_divisible_by_4_probability (n : ℕ) (hn : n = 20) :
  (count_pairs_divisible_by_4 n : ℚ) / (n.choose 2) = 7 / 19 :=
sorry

end NUMINAMATH_CALUDE_product_divisible_by_4_probability_l3528_352871


namespace NUMINAMATH_CALUDE_bike_distance_theorem_l3528_352855

/-- Calculates the distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 3 m/s for 7 seconds covers 21 meters -/
theorem bike_distance_theorem :
  let speed : ℝ := 3
  let time : ℝ := 7
  distance_traveled speed time = 21 := by sorry

end NUMINAMATH_CALUDE_bike_distance_theorem_l3528_352855


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_sphere_volume_ratio_l3528_352865

/-- The ratio of the volume of the inscribed sphere to the volume of the circumscribed sphere of a cube -/
theorem inscribed_circumscribed_sphere_volume_ratio (V₁ V₂ : ℝ) :
  V₁ > 0 →
  V₂ > 0 →
  V₁ = volume_inscribed_sphere_of_cube →
  V₂ = volume_circumscribed_sphere_of_cube →
  V₁ / V₂ = (Real.sqrt 3 / 3) ^ 3 :=
by sorry

/-- The volume of the inscribed sphere of a cube -/
noncomputable def volume_inscribed_sphere_of_cube : ℝ := sorry

/-- The volume of the circumscribed sphere of a cube -/
noncomputable def volume_circumscribed_sphere_of_cube : ℝ := sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_sphere_volume_ratio_l3528_352865


namespace NUMINAMATH_CALUDE_sequence_a_general_term_sequence_b_general_term_l3528_352876

-- Define the sequences
def sequence_a : ℕ → ℕ
  | 1 => 0
  | 2 => 3
  | 3 => 26
  | 4 => 255
  | 5 => 3124
  | _ => 0  -- Default case, not used in the proof

def sequence_b : ℕ → ℕ
  | 1 => 1
  | 2 => 2
  | 3 => 12
  | 4 => 288
  | 5 => 34560
  | _ => 0  -- Default case, not used in the proof

-- Define the general term for sequence a
def general_term_a (n : ℕ) : ℕ := n^n - 1

-- Define the general term for sequence b
def general_term_b (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc * Nat.factorial (i + 1)) 1

-- Theorem for sequence a
theorem sequence_a_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_a n = general_term_a n := by
  sorry

-- Theorem for sequence b
theorem sequence_b_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_b n = general_term_b n := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_general_term_sequence_b_general_term_l3528_352876


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3528_352819

theorem imaginary_part_of_z (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3528_352819


namespace NUMINAMATH_CALUDE_smallest_positive_integer_e_l3528_352824

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -4 ∨ x = 6 ∨ x = 10 ∨ x = -1/2) → 
    e ≤ e') →
  e = 200 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_e_l3528_352824


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l3528_352891

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the point M₀
def M₀ : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem tangent_and_normal_equations :
  let (x₀, y₀) := M₀
  let f' := λ x => 3 * x^2  -- Derivative of f
  let m_tangent := f' x₀    -- Slope of tangent line
  let m_normal := -1 / m_tangent  -- Slope of normal line
  -- Equation of tangent line
  (∀ x y, 12 * x - y - 16 = 0 ↔ y - y₀ = m_tangent * (x - x₀)) ∧
  -- Equation of normal line
  (∀ x y, x + 12 * y - 98 = 0 ↔ y - y₀ = m_normal * (x - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_normal_equations_l3528_352891


namespace NUMINAMATH_CALUDE_age_ratio_l3528_352889

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ  -- Present age of A
  b : ℕ  -- Present age of B

/-- Conditions for the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.a - 10 = (ages.b - 10) / 2) ∧ (ages.a + ages.b = 35)

/-- Theorem stating the ratio of present ages -/
theorem age_ratio (ages : Ages) (h : AgeConditions ages) : 
  (ages.a : ℚ) / ages.b = 3 / 4 := by
  sorry

#check age_ratio

end NUMINAMATH_CALUDE_age_ratio_l3528_352889


namespace NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l3528_352869

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l3528_352869


namespace NUMINAMATH_CALUDE_largest_solution_is_two_l3528_352827

theorem largest_solution_is_two :
  ∃ (x : ℝ), x > 0 ∧ (x / 4 + 2 / (3 * x) = 5 / 6) ∧
  (∀ (y : ℝ), y > 0 → y / 4 + 2 / (3 * y) = 5 / 6 → y ≤ x) ∧
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_is_two_l3528_352827


namespace NUMINAMATH_CALUDE_max_value_xy_l3528_352858

theorem max_value_xy (x y : ℝ) (h1 : x * y + 6 = x + 9 * y) (h2 : y < 1) :
  (∃ (z : ℝ), ∀ (a b : ℝ), a * b + 6 = a + 9 * b → b < 1 → (a + 3) * (b + 1) ≤ z) ∧
  (∃ (x' y' : ℝ), x' * y' + 6 = x' + 9 * y' ∧ y' < 1 ∧ (x' + 3) * (y' + 1) = 27 - 12 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_xy_l3528_352858


namespace NUMINAMATH_CALUDE_square_in_S_l3528_352818

/-- The set S of numbers where n-1, n, and n+1 can be expressed as sums of squares of two positive integers -/
def S : Set ℕ := {n | ∃ a b k l p q : ℕ+, 
  (a^2 + b^2 : ℕ) = n - 1 ∧ 
  (k^2 + l^2 : ℕ) = n ∧ 
  (p^2 + q^2 : ℕ) = n + 1}

/-- If n is in S, then n^2 is also in S -/
theorem square_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_square_in_S_l3528_352818


namespace NUMINAMATH_CALUDE_undefined_expression_expression_undefined_at_nine_l3528_352882

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by sorry

theorem expression_undefined_at_nine : 
  ∃! x : ℝ, x^2 - 18*x + 81 = 0 := by sorry

end NUMINAMATH_CALUDE_undefined_expression_expression_undefined_at_nine_l3528_352882


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3528_352810

def election_votes : List Nat := [1000, 2000, 4000]

theorem winning_candidate_percentage :
  let total_votes := election_votes.sum
  let winning_votes := election_votes.maximum?
  winning_votes.map (λ w => (w : ℚ) / total_votes * 100) = some (4000 / 7000 * 100) := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3528_352810


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_over_sqrt_2_l3528_352823

theorem sqrt_18_minus_sqrt_2_over_sqrt_2 : (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_over_sqrt_2_l3528_352823


namespace NUMINAMATH_CALUDE_laundry_problem_solution_l3528_352817

/-- Represents the laundry shop scenario --/
structure LaundryShop where
  price_per_kilo : ℝ
  kilos_two_days_ago : ℝ
  total_earnings : ℝ

/-- Calculates the total kilos of laundry for three days --/
def total_kilos (shop : LaundryShop) : ℝ :=
  shop.kilos_two_days_ago + 
  (shop.kilos_two_days_ago + 5) + 
  2 * (shop.kilos_two_days_ago + 5)

/-- Theorem stating the solution to the laundry problem --/
theorem laundry_problem_solution (shop : LaundryShop) 
  (h1 : shop.price_per_kilo = 2)
  (h2 : shop.total_earnings = 70) :
  shop.kilos_two_days_ago = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_laundry_problem_solution_l3528_352817


namespace NUMINAMATH_CALUDE_remainder_18_pow_63_mod_5_l3528_352898

theorem remainder_18_pow_63_mod_5 : 18^63 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_18_pow_63_mod_5_l3528_352898


namespace NUMINAMATH_CALUDE_jump_height_to_touch_hoop_l3528_352866

/-- Calculates the jump height needed to touch a basketball hoop -/
theorem jump_height_to_touch_hoop 
  (yao_height_ft : ℕ) 
  (yao_height_in : ℕ) 
  (hoop_height_ft : ℕ) 
  (inches_per_foot : ℕ) : 
  hoop_height_ft * inches_per_foot - (yao_height_ft * inches_per_foot + yao_height_in) = 31 :=
by
  sorry

#check jump_height_to_touch_hoop 7 5 10 12

end NUMINAMATH_CALUDE_jump_height_to_touch_hoop_l3528_352866


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l3528_352853

/-- Calculates the total amount received by a contractor given the contract terms and absences. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_payment := working_days * payment_per_day
  let total_fine := absent_days * fine_per_day
  total_payment - total_fine

/-- Proves that the contractor receives Rs. 425 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_proof_l3528_352853


namespace NUMINAMATH_CALUDE_max_player_salary_l3528_352829

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  ∃ (max_salary : ℕ),
    max_salary = 260000 ∧
    max_salary = total_cap - (n - 1) * min_salary ∧
    max_salary ≥ min_salary ∧
    (n - 1) * min_salary + max_salary ≤ total_cap :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l3528_352829


namespace NUMINAMATH_CALUDE_subset_M_l3528_352861

def P : Set ℝ := {x | 0 ≤ x ∧ x < 1}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def M : Set ℝ := P ∪ Q

theorem subset_M : {0, 2, 3} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l3528_352861


namespace NUMINAMATH_CALUDE_miles_driven_l3528_352846

/-- Calculates the number of miles driven given car rental costs and total expenses --/
theorem miles_driven (rental_cost gas_needed gas_price per_mile_charge total_cost : ℚ) : 
  rental_cost = 150 →
  gas_needed = 8 →
  gas_price = 3.5 →
  per_mile_charge = 0.5 →
  total_cost = 338 →
  (total_cost - (rental_cost + gas_needed * gas_price)) / per_mile_charge = 320 := by
  sorry


end NUMINAMATH_CALUDE_miles_driven_l3528_352846


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l3528_352843

theorem soccer_ball_cost (F S : ℝ) 
  (eq1 : 3 * F + S = 155) 
  (eq2 : 2 * F + 3 * S = 220) : 
  S = 50 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l3528_352843


namespace NUMINAMATH_CALUDE_f_min_value_l3528_352874

/-- The polynomial f(x) defined for a positive integer n and real x -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (Finset.range (2*n+1)).sum (fun k => (2*n+1-k) * x^k)

/-- Theorem stating that the minimum value of f(x) is n+1 and occurs at x = -1 -/
theorem f_min_value (n : ℕ+) :
  (∀ x : ℝ, f n x ≥ f n (-1)) ∧ f n (-1) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3528_352874


namespace NUMINAMATH_CALUDE_agent_007_encryption_possible_l3528_352813

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_agent_007_encryption_possible_l3528_352813


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3528_352884

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I) 
  (h2 : Complex.abs z = Real.sqrt 5) 
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3528_352884


namespace NUMINAMATH_CALUDE_segment_length_l3528_352814

/-- Given two points P and Q on a line segment AB, where:
    - P and Q are on the same side of the midpoint of AB
    - P divides AB in the ratio 3:5
    - Q divides AB in the ratio 4:5
    - PQ = 3
    Prove that the length of AB is 43.2 -/
theorem segment_length (A B P Q : Real) (h1 : P ∈ Set.Icc A B) (h2 : Q ∈ Set.Icc A B)
    (h3 : (P - A) / (B - A) = 3 / 8) (h4 : (Q - A) / (B - A) = 4 / 9) (h5 : Q - P = 3) :
    B - A = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l3528_352814


namespace NUMINAMATH_CALUDE_michelle_oranges_l3528_352836

theorem michelle_oranges :
  ∀ (total : ℕ),
  (total / 3 : ℚ) + 5 + 7 = total →
  total = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_michelle_oranges_l3528_352836


namespace NUMINAMATH_CALUDE_ball_probability_and_replacement_l3528_352897

/-- Given a bag with red, yellow, and blue balls, this theorem proves:
    1. The initial probability of drawing a red ball.
    2. The number of red balls replaced to achieve a specific probability of drawing a yellow ball. -/
theorem ball_probability_and_replacement 
  (initial_red : ℕ) 
  (initial_yellow : ℕ) 
  (initial_blue : ℕ) 
  (replaced : ℕ) :
  initial_red = 10 → 
  initial_yellow = 2 → 
  initial_blue = 8 → 
  (initial_red : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 1/2 ∧
  (initial_yellow + replaced : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 2/5 →
  replaced = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_and_replacement_l3528_352897


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3528_352840

/-- Given Tom's current age t and Sara's current age s, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_problem (t s : ℕ) (h1 : t - 3 = 2 * (s - 3)) (h2 : t - 8 = 3 * (s - 8)) :
  ∃ x : ℕ, x = 7 ∧ (t + x : ℚ) / (s + x) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3528_352840


namespace NUMINAMATH_CALUDE_circle_center_l3528_352831

/-- The center of a circle given by the equation x^2 + 8x + y^2 - 4y = 16 is (-4, 2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 8*x + y^2 - 4*y = 16) → (x + 4)^2 + (y - 2)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3528_352831


namespace NUMINAMATH_CALUDE_impossible_segment_arrangement_l3528_352860

/-- A segment on the number line -/
structure Segment where
  start : ℕ
  length : ℕ
  h1 : start ≥ 1
  h2 : start + length ≤ 100

/-- The set of all possible segments -/
def AllSegments : Set Segment :=
  { s : Segment | s.start ≥ 1 ∧ s.start + s.length ≤ 100 ∧ s.length ∈ Finset.range 51 }

/-- The theorem stating the impossibility of the segment arrangement -/
theorem impossible_segment_arrangement :
  ¬ ∃ (segments : Finset Segment),
    segments.card = 50 ∧
    (∀ s ∈ segments, s ∈ AllSegments) ∧
    (∀ n ∈ Finset.range 51, ∃ s ∈ segments, s.length = n) :=
sorry

end NUMINAMATH_CALUDE_impossible_segment_arrangement_l3528_352860


namespace NUMINAMATH_CALUDE_simplify_expression_l3528_352816

theorem simplify_expression (n : ℕ) : (2^(n+5) - 3*(2^n)) / (3*(2^(n+3))) = 29 / 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3528_352816


namespace NUMINAMATH_CALUDE_passengers_taken_at_first_station_l3528_352820

/-- Represents the number of passengers on the train at various points --/
structure TrainPassengers where
  initial : ℕ
  afterFirstDrop : ℕ
  afterFirstPickup : ℕ
  afterSecondDrop : ℕ
  afterSecondPickup : ℕ
  final : ℕ

/-- Represents the passenger flow on the train's journey --/
def trainJourney (x : ℕ) : TrainPassengers :=
  { initial := 270,
    afterFirstDrop := 270 - (270 / 3),
    afterFirstPickup := 270 - (270 / 3) + x,
    afterSecondDrop := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2),
    afterSecondPickup := (270 - (270 / 3) + x) - ((270 - (270 / 3) + x) / 2) + 12,
    final := 242 }

/-- Theorem stating that 280 passengers were taken at the first station --/
theorem passengers_taken_at_first_station :
  ∃ (x : ℕ), trainJourney x = trainJourney 280 ∧ 
  (trainJourney x).afterSecondPickup = (trainJourney x).final :=
sorry


end NUMINAMATH_CALUDE_passengers_taken_at_first_station_l3528_352820


namespace NUMINAMATH_CALUDE_centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l3528_352850

-- Define basic biological concepts
def Chromosome : Type := Unit
def Centromere : Type := Unit
def Cell : Type := Unit
def Ribosome : Type := Unit
def DNAMolecule : Type := Unit
def Chromatid : Type := Unit
def HomologousChromosome : Type := Unit

-- Define the properties
def has_ribosome (c : Cell) : Prop := sorry
def is_eukaryotic (c : Cell) : Prop := sorry
def number_of_centromeres (c : Cell) : ℕ := sorry
def number_of_chromosomes (c : Cell) : ℕ := sorry
def number_of_dna_molecules (c : Cell) : ℕ := sorry
def number_of_chromatids (c : Cell) : ℕ := sorry
def size_and_shape (h : HomologousChromosome) : ℕ := sorry

-- State the theorem
theorem centromeres_equal_chromosomes :
  ∀ (c : Cell), number_of_centromeres c = number_of_chromosomes c :=
sorry

-- State the conditions
axiom cells_with_ribosomes :
  ∃ (c : Cell), has_ribosome c ∧ ¬is_eukaryotic c

axiom dna_chromatid_ratio :
  ∀ (c : Cell), 
    (number_of_dna_molecules c = number_of_chromatids c) ∨
    (number_of_dna_molecules c = 1 ∧ number_of_chromatids c = 0)

axiom homologous_chromosomes_different :
  ∃ (h1 h2 : HomologousChromosome), size_and_shape h1 ≠ size_and_shape h2

-- The main theorem stating that the statement is false
theorem centromeres_necessarily_equal_chromosomes :
  ¬(∃ (c : Cell), number_of_centromeres c ≠ number_of_chromosomes c) :=
sorry

end NUMINAMATH_CALUDE_centromeres_equal_chromosomes_centromeres_necessarily_equal_chromosomes_l3528_352850


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3528_352854

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3528_352854


namespace NUMINAMATH_CALUDE_parabolas_intersection_circle_l3528_352878

/-- The parabolas y = (x + 2)^2 and x + 8 = (y - 2)^2 intersect at four points that lie on a circle with radius squared equal to 4 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x + 2)^2 ∧ x + 8 = (y - 2)^2) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_circle_l3528_352878


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3528_352886

/-- The number of possible six-ring arrangements on four fingers -/
def ring_arrangements : ℕ := 618854400

/-- The number of distinguishable rings -/
def total_rings : ℕ := 10

/-- The number of rings to be arranged -/
def arranged_rings : ℕ := 6

/-- The number of fingers (excluding thumb) -/
def fingers : ℕ := 4

theorem ring_arrangement_count :
  ring_arrangements = (total_rings.choose arranged_rings) * (arranged_rings.factorial) * (fingers ^ arranged_rings) :=
sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3528_352886


namespace NUMINAMATH_CALUDE_exists_noncommuting_matrix_exp_l3528_352863

open Matrix

/-- Definition of matrix exponential -/
def matrix_exp (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  1 + M + (1/2) • (M * M) + (1/6) • (M * M * M) + sorry

/-- Theorem: There exist 2x2 matrices A and B such that exp(A+B) ≠ exp(A)exp(B) -/
theorem exists_noncommuting_matrix_exp :
  ∃ (A B : Matrix (Fin 2) (Fin 2) ℝ), matrix_exp (A + B) ≠ matrix_exp A * matrix_exp B :=
sorry

end NUMINAMATH_CALUDE_exists_noncommuting_matrix_exp_l3528_352863


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l3528_352894

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l3528_352894


namespace NUMINAMATH_CALUDE_tan_A_value_l3528_352868

theorem tan_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 4 * Real.sin A ^ 2 - 4 * Real.sin A * Real.cos A + Real.cos A ^ 2 = 0) : 
  Real.tan A = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tan_A_value_l3528_352868


namespace NUMINAMATH_CALUDE_complexity_theorem_l3528_352833

-- Define complexity of a positive integer
def complexity (n : ℕ) : ℕ := sorry

-- Define the property for part (a)
def property_a (n : ℕ) : Prop :=
  ∀ m : ℕ, n ≤ m → m ≤ 2*n → complexity m ≤ complexity n

-- Define the property for part (b)
def property_b (n : ℕ) : Prop :=
  ∀ m : ℕ, n < m → m < 2*n → complexity m < complexity n

theorem complexity_theorem :
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n = 2^k → property_a n) ∧
  (¬ ∃ n : ℕ, n > 1 ∧ property_b n) := by sorry

end NUMINAMATH_CALUDE_complexity_theorem_l3528_352833


namespace NUMINAMATH_CALUDE_calculate_expression_l3528_352877

theorem calculate_expression : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_calculate_expression_l3528_352877


namespace NUMINAMATH_CALUDE_factorization_equality_l3528_352895

theorem factorization_equality (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3528_352895


namespace NUMINAMATH_CALUDE_rye_flour_amount_l3528_352842

/-- The amount of rye flour Sarah bought -/
def rye_flour : ℝ := sorry

/-- The amount of whole-wheat bread flour Sarah bought -/
def whole_wheat_bread : ℝ := 10

/-- The amount of chickpea flour Sarah bought -/
def chickpea : ℝ := 3

/-- The amount of whole-wheat pastry flour Sarah had at home -/
def whole_wheat_pastry : ℝ := 2

/-- The total amount of flour Sarah has now -/
def total_flour : ℝ := 20

/-- Theorem stating that the amount of rye flour Sarah bought is 5 pounds -/
theorem rye_flour_amount : rye_flour = 5 := by
  sorry

end NUMINAMATH_CALUDE_rye_flour_amount_l3528_352842


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3528_352822

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 61) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : chemistry = 67) 
  (h5 : average = 72) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 85 := by
sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3528_352822


namespace NUMINAMATH_CALUDE_total_cars_l3528_352873

/-- The number of cars owned by five people given specific relationships between their car counts -/
theorem total_cars (tommy : ℕ) (jessie : ℕ) : 
  tommy = 7 →
  jessie = 9 →
  (tommy + jessie + (jessie + 2) + (tommy - 3) + 2 * (jessie + 2)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_l3528_352873


namespace NUMINAMATH_CALUDE_equal_goldfish_theorem_l3528_352864

/-- Number of months for Brent and Gretel to have the same number of goldfish -/
def equal_goldfish_months : ℕ := 8

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 3

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 243

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 3

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

/-- Theorem stating that Brent and Gretel have the same number of goldfish after equal_goldfish_months -/
theorem equal_goldfish_theorem : 
  brent_goldfish equal_goldfish_months = gretel_goldfish equal_goldfish_months :=
sorry

end NUMINAMATH_CALUDE_equal_goldfish_theorem_l3528_352864


namespace NUMINAMATH_CALUDE_sixth_equation_pattern_l3528_352808

/-- The sum of n consecutive odd numbers starting from a given odd number -/
def sum_consecutive_odds (start : ℕ) (n : ℕ) : ℕ :=
  (start + n - 1) * n

/-- The nth cube -/
def cube (n : ℕ) : ℕ := n^3

theorem sixth_equation_pattern : sum_consecutive_odds 31 6 = cube 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_equation_pattern_l3528_352808


namespace NUMINAMATH_CALUDE_probability_of_more_than_five_draws_l3528_352849

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ := 20 / 63

theorem probability_of_more_than_five_draws :
  let total_combinations := Nat.choose total_pennies shiny_pennies
  let favorable_combinations := Nat.choose 5 3 * Nat.choose 4 1
  (favorable_combinations : ℚ) / total_combinations = probability_more_than_five_draws :=
sorry

end NUMINAMATH_CALUDE_probability_of_more_than_five_draws_l3528_352849


namespace NUMINAMATH_CALUDE_one_rupee_coins_count_l3528_352872

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in paise -/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | CoinType.OneRupee => 100
  | CoinType.FiftyPaise => 50
  | CoinType.TwentyFivePaise => 25

/-- The total value of all coins in the bag in paise -/
def totalValue : ℕ := 105 * 100

/-- The number of each type of coin in the bag -/
def numEachCoin : ℕ := 60

/-- The total number of coins in the bag -/
def totalCoins : ℕ := 3 * numEachCoin

theorem one_rupee_coins_count :
  ∃ (n : ℕ), n = numEachCoin ∧
    n * coinValue CoinType.OneRupee +
    n * coinValue CoinType.FiftyPaise +
    n * coinValue CoinType.TwentyFivePaise = totalValue ∧
    3 * n = totalCoins := by
  sorry

end NUMINAMATH_CALUDE_one_rupee_coins_count_l3528_352872


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l3528_352899

theorem revenue_change_after_price_and_quantity_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percentage : ℝ) 
  (quantity_decrease_percentage : ℝ) :
  let new_price := original_price * (1 + price_increase_percentage)
  let new_quantity := original_quantity * (1 - quantity_decrease_percentage)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  price_increase_percentage = 0.7 →
  quantity_decrease_percentage = 0.2 →
  (new_revenue - original_revenue) / original_revenue = 0.36 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_change_l3528_352899


namespace NUMINAMATH_CALUDE_product_of_numbers_l3528_352815

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3528_352815


namespace NUMINAMATH_CALUDE_original_line_length_l3528_352879

-- Define the units
def cm : ℝ := 1
def meter : ℝ := 100 * cm

-- Define the problem parameters
def erased_length : ℝ := 10 * cm
def remaining_length : ℝ := 90 * cm

-- State the theorem
theorem original_line_length :
  ∃ (original_length : ℝ),
    original_length = remaining_length + erased_length ∧
    original_length = 1 * meter :=
by sorry

end NUMINAMATH_CALUDE_original_line_length_l3528_352879


namespace NUMINAMATH_CALUDE_production_line_b_units_l3528_352837

theorem production_line_b_units (total : ℕ) (a b c : ℕ) : 
  total = 16800 →
  total = a + b + c →
  b - a = c - b →
  b = 5600 := by
  sorry

end NUMINAMATH_CALUDE_production_line_b_units_l3528_352837


namespace NUMINAMATH_CALUDE_unique_max_sum_pair_l3528_352859

theorem unique_max_sum_pair :
  ∃! (x y : ℕ), 
    (∃ (k : ℕ), 19 * x + 95 * y = k * k) ∧
    19 * x + 95 * y ≤ 1995 ∧
    (∀ (a b : ℕ), (∃ (m : ℕ), 19 * a + 95 * b = m * m) → 
      19 * a + 95 * b ≤ 1995 → 
      a + b ≤ x + y) :=
by sorry

end NUMINAMATH_CALUDE_unique_max_sum_pair_l3528_352859


namespace NUMINAMATH_CALUDE_min_area_of_two_squares_l3528_352848

/-- Given a wire of length 20 cm cut into two parts, with each part forming a square 
    where the part's length is the square's perimeter, the minimum combined area 
    of the two squares is 12.5 square centimeters. -/
theorem min_area_of_two_squares (x : ℝ) : 
  0 ≤ x → 
  x ≤ 20 → 
  (x^2 / 16 + (20 - x)^2 / 16) ≥ 12.5 := by
  sorry

end NUMINAMATH_CALUDE_min_area_of_two_squares_l3528_352848


namespace NUMINAMATH_CALUDE_pizza_size_increase_l3528_352896

theorem pizza_size_increase (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt 1.21
  (R ^ 2 - r ^ 2) / r ^ 2 = 0.21000000000000018 →
  (R - r) / r = 0.1 := by
sorry

end NUMINAMATH_CALUDE_pizza_size_increase_l3528_352896


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3528_352857

-- Define the number of knights
def total_knights : ℕ := 30

-- Define the number of knights chosen
def chosen_knights : ℕ := 4

-- Function to calculate the probability
def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (total_knights - chosen_knights + 1) (chosen_knights - 1) : ℚ) / 
      (Nat.choose total_knights chosen_knights : ℚ)

-- Theorem statement
theorem adjacent_knights_probability :
  probability_adjacent_knights = 4961 / 5481 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3528_352857


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3528_352893

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3528_352893


namespace NUMINAMATH_CALUDE_power_function_through_point_l3528_352883

theorem power_function_through_point (a : ℝ) : 
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3528_352883


namespace NUMINAMATH_CALUDE_expected_original_positions_value_l3528_352867

/-- Represents the number of balls in the circle -/
def num_balls : ℕ := 7

/-- Represents the probability of a ball being in its original position after two transpositions -/
def prob_original_position : ℚ := 9 / 14

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original_positions : ℚ := num_balls * prob_original_position

theorem expected_original_positions_value :
  expected_original_positions = 4.5 := by sorry

end NUMINAMATH_CALUDE_expected_original_positions_value_l3528_352867


namespace NUMINAMATH_CALUDE_inverse_proposition_is_false_l3528_352830

theorem inverse_proposition_is_false : ¬∀ a : ℝ, (abs a = abs 6) → (a = 6) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_is_false_l3528_352830


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3528_352888

theorem basketball_free_throws (total_score : ℕ) (three_point_shots : ℕ) 
  (h1 : total_score = 79)
  (h2 : 3 * three_point_shots = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h3 : free_throws = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h4 : three_point_shots = 4) :
  free_throws = 12 :=
by
  sorry

#check basketball_free_throws

end NUMINAMATH_CALUDE_basketball_free_throws_l3528_352888


namespace NUMINAMATH_CALUDE_platform_length_is_605_l3528_352811

/-- Calculates the length of a platform given train movement parameters. -/
def platformLength (
  platformPassTime : Real
) (manPassTime : Real)
  (manDistance : Real)
  (initialSpeed : Real)
  (acceleration : Real) : Real :=
  let trainLength := manPassTime * initialSpeed + 0.5 * acceleration * manPassTime ^ 2 - manDistance
  let platformPassDistance := platformPassTime * initialSpeed + 0.5 * acceleration * platformPassTime ^ 2
  platformPassDistance - trainLength

/-- The length of the platform is 605 meters. -/
theorem platform_length_is_605 :
  platformLength 40 20 5 15 0.5 = 605 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_is_605_l3528_352811


namespace NUMINAMATH_CALUDE_g_13_l3528_352803

def g (n : ℕ) : ℕ := n^2 + 2*n + 41

theorem g_13 : g 13 = 236 := by
  sorry

end NUMINAMATH_CALUDE_g_13_l3528_352803


namespace NUMINAMATH_CALUDE_train_length_l3528_352835

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 180 → time_s = 18 → speed_kmh * (1000 / 3600) * time_s = 900 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3528_352835


namespace NUMINAMATH_CALUDE_square_diff_cube_seven_six_l3528_352800

theorem square_diff_cube_seven_six : (7^2 - 6^2)^3 = 2197 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cube_seven_six_l3528_352800


namespace NUMINAMATH_CALUDE_cupcake_problem_l3528_352839

theorem cupcake_problem (cupcake_cost : ℚ) (individual_payment : ℚ) :
  cupcake_cost = 3/2 →
  individual_payment = 9 →
  (2 * individual_payment) / cupcake_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_problem_l3528_352839


namespace NUMINAMATH_CALUDE_factorial_division_l3528_352801

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3528_352801


namespace NUMINAMATH_CALUDE_hotel_arrangement_l3528_352844

/-- The number of ways to distribute n distinct objects into k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition n distinct objects into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem hotel_arrangement :
  distribute 5 3 = 150 :=
by
  -- Define distribute in terms of stirling2 and factorial
  have h1 : ∀ n k, distribute n k = stirling2 n k * Nat.factorial k
  sorry
  
  -- Use the specific values for our problem
  have h2 : stirling2 5 3 = 25
  sorry
  
  -- Apply the definitions and properties
  rw [h1]
  simp [h2]
  -- The proof is completed by computation
  sorry

end NUMINAMATH_CALUDE_hotel_arrangement_l3528_352844


namespace NUMINAMATH_CALUDE_earth_fresh_water_coverage_l3528_352880

theorem earth_fresh_water_coverage : 
  ∀ (land_coverage : ℝ) (salt_water_percentage : ℝ),
  land_coverage = 3 / 10 →
  salt_water_percentage = 97 / 100 →
  (1 - land_coverage) * (1 - salt_water_percentage) = 21 / 1000 := by
sorry

end NUMINAMATH_CALUDE_earth_fresh_water_coverage_l3528_352880


namespace NUMINAMATH_CALUDE_shipment_average_weight_l3528_352804

/-- Represents the weight distribution of boxes in a shipment. -/
structure Shipment where
  total_boxes : ℕ
  light_boxes : ℕ
  heavy_boxes : ℕ
  light_weight : ℕ
  heavy_weight : ℕ

/-- Calculates the average weight of boxes after removing some heavy boxes. -/
def new_average (s : Shipment) (removed : ℕ) : ℚ :=
  (s.light_boxes * s.light_weight + (s.heavy_boxes - removed) * s.heavy_weight) /
  (s.light_boxes + s.heavy_boxes - removed)

/-- Theorem stating the average weight of boxes in the shipment. -/
theorem shipment_average_weight (s : Shipment) :
  s.total_boxes = 20 ∧
  s.light_weight = 10 ∧
  s.heavy_weight = 20 ∧
  s.light_boxes + s.heavy_boxes = s.total_boxes ∧
  new_average s 10 = 16 →
  (s.light_boxes * s.light_weight + s.heavy_boxes * s.heavy_weight) / s.total_boxes = 39/2 := by
  sorry

#check shipment_average_weight

end NUMINAMATH_CALUDE_shipment_average_weight_l3528_352804


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3528_352825

theorem three_numbers_problem :
  ∃ (a b c : ℕ),
    (Nat.gcd a b = 8) ∧
    (Nat.gcd b c = 2) ∧
    (Nat.gcd a c = 6) ∧
    (Nat.lcm (Nat.lcm a b) c = 1680) ∧
    (max a (max b c) > 100) ∧
    (max a (max b c) ≤ 200) ∧
    ((∃ n : ℕ, a = n^4) ∨ (∃ n : ℕ, b = n^4) ∨ (∃ n : ℕ, c = n^4)) ∧
    ((a = 120 ∧ b = 16 ∧ c = 42) ∨ (a = 168 ∧ b = 16 ∧ c = 30)) :=
by
  sorry

#check three_numbers_problem

end NUMINAMATH_CALUDE_three_numbers_problem_l3528_352825


namespace NUMINAMATH_CALUDE_sector_angle_l3528_352832

/-- A circular sector with area 1 cm² and perimeter 4 cm has a central angle of 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  (1/2 * α * r^2 = 1) → (2*r + α*r = 4) → α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3528_352832


namespace NUMINAMATH_CALUDE_savings_proof_l3528_352851

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the person's savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end NUMINAMATH_CALUDE_savings_proof_l3528_352851


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3528_352812

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 4 * a + b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + y = x * y → a + b ≤ x + y ∧ a + b = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3528_352812


namespace NUMINAMATH_CALUDE_eggs_leftover_l3528_352834

def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + emma_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l3528_352834


namespace NUMINAMATH_CALUDE_office_clerks_count_l3528_352862

/-- Calculates the number of clerks in an office given specific salary information. -/
theorem office_clerks_count (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (officer_count : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  officer_count = 2 →
  ∃ (clerk_count : ℕ), 
    (officer_count * officer_avg + clerk_count * clerk_avg) / (officer_count + clerk_count) = total_avg ∧
    clerk_count = 170 :=
by sorry

end NUMINAMATH_CALUDE_office_clerks_count_l3528_352862


namespace NUMINAMATH_CALUDE_income_education_relationship_l3528_352875

/-- Represents the linear regression model for annual income and educational expenditure -/
structure IncomeEducationModel where
  -- x: annual income in ten thousand yuan
  -- y: annual educational expenditure in ten thousand yuan
  slope : Real
  intercept : Real
  equation : Real → Real := λ x => slope * x + intercept

/-- Theorem: In the given linear regression model, an increase of 1 in income
    results in an increase of 0.15 in educational expenditure -/
theorem income_education_relationship (model : IncomeEducationModel)
    (h_slope : model.slope = 0.15)
    (h_intercept : model.intercept = 0.2) :
    ∀ x : Real, model.equation (x + 1) - model.equation x = 0.15 := by
  sorry

#check income_education_relationship

end NUMINAMATH_CALUDE_income_education_relationship_l3528_352875


namespace NUMINAMATH_CALUDE_fred_red_marbles_l3528_352845

/-- Fred's marble collection --/
structure MarbleCollection where
  total : ℕ
  darkBlue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Theorem: Fred has 60 red marbles --/
theorem fred_red_marbles (m : MarbleCollection) : m.red = 60 :=
  by
  have h1 : m.total = 120 := by sorry
  have h2 : m.darkBlue = m.total / 4 := by sorry
  have h3 : m.red = 2 * m.darkBlue := by sorry
  have h4 : m.green = 10 := by sorry
  have h5 : m.yellow = 5 := by sorry
  
  -- Proof
  sorry


end NUMINAMATH_CALUDE_fred_red_marbles_l3528_352845


namespace NUMINAMATH_CALUDE_f_of_2_equals_2_l3528_352852

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

-- State the theorem
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_2_l3528_352852


namespace NUMINAMATH_CALUDE_ln_square_plus_ln_inequality_l3528_352870

theorem ln_square_plus_ln_inequality (x : ℝ) :
  (Real.log x)^2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_square_plus_ln_inequality_l3528_352870


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l3528_352805

theorem distinct_primes_in_product : ∃ (S : Finset Nat), 
  (∀ p ∈ S, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (85 * 87 * 90 * 92) ↔ p ∈ S)) ∧ 
  Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l3528_352805


namespace NUMINAMATH_CALUDE_book_length_problem_l3528_352887

/-- Represents the problem of determining book lengths based on reading rates and times -/
theorem book_length_problem (book1_pages book2_pages_read : ℕ) 
  (book1_rate book2_rate : ℕ) (h1 : book1_rate = 40) (h2 : book2_rate = 60) :
  (2 * book1_pages / 3 = book1_pages / 3 + 30) →
  (2 * book1_pages / (3 * book1_rate) = book2_pages_read / book2_rate) →
  book1_pages = 90 ∧ book2_pages_read = 45 := by
  sorry

#check book_length_problem

end NUMINAMATH_CALUDE_book_length_problem_l3528_352887


namespace NUMINAMATH_CALUDE_triangle_inequality_l3528_352838

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  Real.sin t.A * Real.cos t.C + t.A * Real.cos t.B > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3528_352838


namespace NUMINAMATH_CALUDE_intersection_points_on_hyperbola_l3528_352892

/-- The intersection points of the lines 2tx - 3y - 5t = 0 and x - 3ty + 5 = 0,
    where t is a real number, lie on a hyperbola. -/
theorem intersection_points_on_hyperbola :
  ∀ (t x y : ℝ),
    (2 * t * x - 3 * y - 5 * t = 0) →
    (x - 3 * t * y + 5 = 0) →
    ∃ (a b : ℝ), x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_on_hyperbola_l3528_352892


namespace NUMINAMATH_CALUDE_kevins_stamps_l3528_352847

theorem kevins_stamps (carl_stamps : ℕ) (difference : ℕ) (h1 : carl_stamps = 89) (h2 : difference = 32) :
  carl_stamps - difference = 57 := by
  sorry

end NUMINAMATH_CALUDE_kevins_stamps_l3528_352847
