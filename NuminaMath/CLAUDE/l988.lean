import Mathlib

namespace arithmetic_sequence_sum_l988_98836

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end arithmetic_sequence_sum_l988_98836


namespace triangle_exists_for_all_x_l988_98806

/-- Represents an equilateral triangle with points D, E, F on its sides -/
structure TriangleWithPoints where
  -- Side length of the equilateral triangle
  side : ℝ
  -- Position of point D on side BC
  d : ℝ
  -- Position of point E on side CA
  e : ℝ
  -- Position of point F on side AB
  f : ℝ
  -- Ensure D is on side BC
  h_d : d ≥ 0 ∧ d ≤ side
  -- Ensure E is on side CA
  h_e : e ≥ 0 ∧ e ≤ side
  -- Ensure F is on side AB
  h_f : f ≥ 0 ∧ f ≤ side
  -- Ensure D, E, F form a straight line
  h_straight : d + e + f = side

/-- The main theorem stating that for any real x, there exists a valid triangle configuration -/
theorem triangle_exists_for_all_x (x : ℝ) : 
  ∃ t : TriangleWithPoints, 
    t.d = 4 ∧ 
    t.side - t.d = 2*x ∧ 
    t.e = x + 5 ∧ 
    t.side - t.e - t.f = 3 ∧ 
    t.f = 7 + x :=
  sorry

end triangle_exists_for_all_x_l988_98806


namespace function_composition_sum_l988_98893

theorem function_composition_sum (a b : ℝ) :
  (∀ x, (5 * (a * x + b) - 7) = 4 * x + 6) →
  a + b = 17 / 5 := by
sorry

end function_composition_sum_l988_98893


namespace absolute_value_simplification_l988_98839

theorem absolute_value_simplification : |(-5^2 + 6 * 2)| = 13 := by
  sorry

end absolute_value_simplification_l988_98839


namespace smallest_n_with_abc_property_l988_98874

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n + 1) → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ m < 96, ¬(has_abc_property m)) ∧ has_abc_property 96 :=
sorry

end smallest_n_with_abc_property_l988_98874


namespace prob_select_seventh_grade_prob_select_one_from_each_grade_l988_98835

structure School :=
  (seventh_grade : Finset Nat)
  (eighth_grade : Finset Nat)
  (h1 : seventh_grade.card = 2)
  (h2 : eighth_grade.card = 2)
  (h3 : seventh_grade ∩ eighth_grade = ∅)

def total_students (s : School) : Finset Nat :=
  s.seventh_grade ∪ s.eighth_grade

theorem prob_select_seventh_grade (s : School) :
  (s.seventh_grade.card : ℚ) / (total_students s).card = 1 / 2 := by sorry

theorem prob_select_one_from_each_grade (s : School) :
  let total_pairs := (total_students s).card.choose 2
  let mixed_pairs := s.seventh_grade.card * s.eighth_grade.card * 2
  (mixed_pairs : ℚ) / total_pairs = 2 / 3 := by sorry

end prob_select_seventh_grade_prob_select_one_from_each_grade_l988_98835


namespace small_boxes_count_l988_98830

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 504) 
  (h2 : chocolates_per_box = 28) : 
  total_chocolates / chocolates_per_box = 18 := by
  sorry

#check small_boxes_count

end small_boxes_count_l988_98830


namespace probability_bounds_l988_98849

theorem probability_bounds (n : ℕ) (m₀ : ℕ) (p : ℝ) 
  (h_n : n = 120) 
  (h_m₀ : m₀ = 32) 
  (h_most_probable : m₀ = ⌊n * p + 0.5⌋) : 
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end probability_bounds_l988_98849


namespace sunlovers_happy_days_l988_98847

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end sunlovers_happy_days_l988_98847


namespace count_less_than_one_l988_98807

def number_list : List ℝ := [0.03, 1.5, -0.2, 0.76]

theorem count_less_than_one : 
  (number_list.filter (λ x => x < 1)).length = 3 := by
  sorry

end count_less_than_one_l988_98807


namespace correct_calculation_l988_98878

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end correct_calculation_l988_98878


namespace average_side_length_of_squares_l988_98860

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end average_side_length_of_squares_l988_98860


namespace mark_sold_nine_boxes_l988_98855

/-- Given that Mark and Ann were allocated n boxes of cookies to sell, prove that Mark sold 9 boxes. -/
theorem mark_sold_nine_boxes (n : ℕ) (mark_boxes ann_boxes : ℕ) : 
  n = 10 →
  mark_boxes < n →
  ann_boxes = n - 2 →
  mark_boxes ≥ 1 →
  ann_boxes ≥ 1 →
  mark_boxes + ann_boxes < n →
  mark_boxes = 9 := by
sorry

end mark_sold_nine_boxes_l988_98855


namespace f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l988_98879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity (a : ℝ) :
  (a > 0 → ∀ x y, x > Real.log a → y > Real.log a → x < y → f a x < f a y) ∧
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) :=
sorry

theorem f_monotonic_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

theorem f_monotonic_decreasing_increasing_iff (a : ℝ) :
  (∀ x y, x < y → x ≤ 0 → f a x > f a y) ∧
  (∀ x y, x < y → x ≥ 0 → f a x < f a y) ↔
  a = 1 :=
sorry

end f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l988_98879


namespace geometric_sequence_tan_value_l988_98870

/-- Given a geometric sequence {a_n} where a₁a₁₃ + 2a₇² = 4π, prove that tan(a₂a₁₂) = √3 -/
theorem geometric_sequence_tan_value (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_sum : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi)  -- Given equation
  : Real.tan (a 2 * a 12) = Real.sqrt 3 := by
  sorry

end geometric_sequence_tan_value_l988_98870


namespace triangle_circle_area_relation_l988_98812

theorem triangle_circle_area_relation (A B C : ℝ) : 
  -- The triangle is inscribed in a circle
  -- The triangle has side lengths of 20, 21, and 29
  -- A, B, and C are the areas of the three parts outside the triangle
  -- C is the largest area among A, B, and C
  (20 : ℝ)^2 + 21^2 = 29^2 →  -- This ensures it's a right triangle
  A ≥ 0 → B ≥ 0 → C ≥ 0 →
  C ≥ A → C ≥ B →
  -- Prove the relation
  A + B + 210 = C := by
  sorry

end triangle_circle_area_relation_l988_98812


namespace arithmetic_sequence_property_l988_98895

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_a6 : a 6 = 7) :
  a 5 = 15 :=
sorry

end arithmetic_sequence_property_l988_98895


namespace bus_seating_capacity_l988_98892

theorem bus_seating_capacity : 
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 11
  
  left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity = 92 :=
by sorry

end bus_seating_capacity_l988_98892


namespace max_intersections_circles_lines_l988_98803

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to count the number of intersection points -/
def count_intersections (circles : List Circle) (lines : List Line) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem max_intersections_circles_lines :
  ∀ (circles : List Circle) (lines : List Line),
    circles.length = 2 →
    lines.length = 3 →
    (∃ (l : Line) (c : Circle), l ∈ lines ∧ c ∈ circles ∧
      (∀ (c' : Circle) (l' : Line), c' ∈ circles → l' ∈ lines →
        c' ≠ c → l' ≠ l → ¬ (count_intersections [c'] [l'] > 0))) →
    count_intersections circles lines ≤ 12 :=
  sorry

end max_intersections_circles_lines_l988_98803


namespace inscribed_squares_ratio_l988_98823

/-- A right triangle with sides 6, 8, and 10 (hypotenuse) -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1 = 6 ∧ side2 = 8 ∧ hypotenuse = 10

/-- A square inscribed in the triangle with one vertex at the right angle -/
def inscribed_square_at_right_angle (t : RightTriangle) (x : ℝ) : Prop :=
  0 < x ∧ x < t.side1 ∧ x < t.side2

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  0 < y ∧ y < t.side1 ∧ y < t.side2

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_at_right_angle t x)
  (h2 : inscribed_square_on_hypotenuse t y) :
  x / y = 9 / 16 := by
  sorry

end inscribed_squares_ratio_l988_98823


namespace shirt_cost_l988_98814

theorem shirt_cost (initial_amount : ℕ) (change : ℕ) (shirt_cost : ℕ) : 
  initial_amount = 50 → change = 23 → shirt_cost = initial_amount - change → shirt_cost = 27 := by
sorry

end shirt_cost_l988_98814


namespace intersection_nonempty_condition_l988_98867

theorem intersection_nonempty_condition (m n : ℝ) :
  let A : Set ℝ := {x | m - 1 < x ∧ x < m + 1}
  let B : Set ℝ := {x | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ (2 < m + n ∧ m + n < 5) := by sorry

end intersection_nonempty_condition_l988_98867


namespace min_value_is_three_l988_98845

/-- A quadratic function f(x) = ax² + bx + c where b > a and f(x) ≥ 0 for all x ∈ ℝ -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : b > a
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The minimum value of (a+b+c)/(b-a) for a QuadraticFunction is 3 -/
theorem min_value_is_three (f : QuadraticFunction) : 
  (∀ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) ≥ 3) ∧ 
  (∃ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) = 3) := by
  sorry

end min_value_is_three_l988_98845


namespace base_prime_rep_132_l988_98877

def base_prime_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_prime_rep_132 :
  base_prime_representation 132 = [2, 1, 0, 1] :=
by
  sorry

end base_prime_rep_132_l988_98877


namespace divisibility_implies_sum_representation_l988_98890

theorem divisibility_implies_sum_representation (n k : ℕ) 
  (h1 : n > 20) 
  (h2 : k > 1) 
  (h3 : k^2 ∣ n) : 
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end divisibility_implies_sum_representation_l988_98890


namespace no_consecutive_integers_without_real_solutions_l988_98841

theorem no_consecutive_integers_without_real_solutions :
  ¬ ∃ (b c : ℕ), 
    c = b + 1 ∧ 
    b > 0 ∧
    b^2 < 4*c ∧
    c^2 < 4*b :=
by sorry

end no_consecutive_integers_without_real_solutions_l988_98841


namespace f_difference_at_3_and_neg_3_l988_98888

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 636 := by
  sorry

end f_difference_at_3_and_neg_3_l988_98888


namespace range_of_a_l988_98837

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end range_of_a_l988_98837


namespace tank_emptying_time_l988_98858

/-- Proves the time to empty a tank with given conditions -/
theorem tank_emptying_time 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : inlet_rate_per_minute = 3) : 
  (tank_capacity / (tank_capacity / leak_empty_time - inlet_rate_per_minute * 60)) = 8 := by
  sorry

#check tank_emptying_time

end tank_emptying_time_l988_98858


namespace mystic_aquarium_fish_duration_l988_98899

/-- The number of weeks that a given number of fish buckets will last at the Mystic Aquarium -/
def weeks_of_fish (total_buckets : ℕ) : ℕ :=
  let sharks_daily := 4
  let dolphins_daily := sharks_daily / 2
  let others_daily := sharks_daily * 5
  let daily_consumption := sharks_daily + dolphins_daily + others_daily
  let weekly_consumption := daily_consumption * 7
  total_buckets / weekly_consumption

/-- Theorem stating that 546 buckets of fish will last for 3 weeks -/
theorem mystic_aquarium_fish_duration : weeks_of_fish 546 = 3 := by
  sorry

end mystic_aquarium_fish_duration_l988_98899


namespace divisibility_condition_l988_98848

theorem divisibility_condition (x y : ℕ+) :
  (∃ (k : ℕ+), k * (2 * x + 7 * y) = 7 * x + 2 * y) ↔
  (∃ (a : ℕ+), (x = a ∧ y = a) ∨ (x = 4 * a ∧ y = a) ∨ (x = 19 * a ∧ y = a)) :=
by sorry

end divisibility_condition_l988_98848


namespace quadratic_roots_d_value_l988_98809

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 := by
sorry

end quadratic_roots_d_value_l988_98809


namespace like_terms_imply_exponent_one_l988_98802

theorem like_terms_imply_exponent_one (a b : ℝ) (m n x : ℕ) :
  (∃ (k : ℝ), 2 * a^x * b^(n+1) = k * (-3 * a * b^(2*m))) →
  (2*m - n)^x = 1 := by
  sorry

end like_terms_imply_exponent_one_l988_98802


namespace inequality_proof_l988_98898

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x - a| + 1) →
  (Set.Icc 0 2 = {x | f x ≤ 2}) →
  m > 0 →
  n > 0 →
  1/m + 1/n = a →
  m + 2*n ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end inequality_proof_l988_98898


namespace right_triangle_check_triangle_sets_check_l988_98822

theorem right_triangle_check (a b c : ℝ) : Prop :=
  (a * a + b * b = c * c) ∨ (a * a + c * c = b * b) ∨ (b * b + c * c = a * a)

theorem triangle_sets_check : 
  right_triangle_check 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  right_triangle_check 6 8 10 ∧
  right_triangle_check 5 12 13 ∧
  ¬(right_triangle_check (Real.sqrt 3) 2 (Real.sqrt 5)) := by
sorry

end right_triangle_check_triangle_sets_check_l988_98822


namespace product_of_numbers_l988_98813

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 220) : x * y = 56 := by
  sorry

end product_of_numbers_l988_98813


namespace greatest_power_of_two_l988_98876

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, (10^1004 - 4^502) = k * 2^1007) ∧ 
  (∀ m : ℕ, m > 1007 → ¬(∃ k : ℕ, (10^1004 - 4^502) = k * 2^m)) :=
sorry

end greatest_power_of_two_l988_98876


namespace basketball_team_wins_l988_98815

theorem basketball_team_wins (total_games : ℕ) (win_loss_difference : ℕ) 
  (h1 : total_games = 62) 
  (h2 : win_loss_difference = 28) : 
  let games_won := (total_games + win_loss_difference) / 2
  games_won = 45 := by
  sorry

end basketball_team_wins_l988_98815


namespace min_digits_theorem_l988_98825

/-- The minimum number of digits to the right of the decimal point needed to express the given fraction as a decimal -/
def min_decimal_digits : ℕ := 30

/-- The numerator of the fraction -/
def numerator : ℕ := 987654321

/-- The denominator of the fraction -/
def denominator : ℕ := 2^30 * 5^6

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction numerator/denominator as a decimal is min_decimal_digits -/
theorem min_digits_theorem :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, m * denominator ≠ numerator * 10^n) ∧
  (∃ m : ℕ, m * denominator = numerator * 10^min_decimal_digits) :=
sorry

end min_digits_theorem_l988_98825


namespace no_solution_for_equation_l988_98817

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) :=
by sorry

end no_solution_for_equation_l988_98817


namespace three_possible_medians_l988_98869

-- Define the set of game scores
def gameScores (x y : ℤ) : Finset ℤ := {x, 11, 13, y, 12}

-- Define the median of a set of integers
def median (s : Finset ℤ) : ℤ := sorry

-- Theorem statement
theorem three_possible_medians :
  ∃ (m₁ m₂ m₃ : ℤ), ∀ (x y : ℤ),
    (∃ (m : ℤ), median (gameScores x y) = m) →
    (m = m₁ ∨ m = m₂ ∨ m = m₃) ∧
    (m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃) :=
  sorry

#check three_possible_medians

end three_possible_medians_l988_98869


namespace machinery_cost_proof_l988_98832

def total_amount : ℝ := 250
def raw_materials_cost : ℝ := 100
def cash_percentage : ℝ := 0.1

def machinery_cost : ℝ := total_amount - raw_materials_cost - (cash_percentage * total_amount)

theorem machinery_cost_proof : machinery_cost = 125 := by
  sorry

end machinery_cost_proof_l988_98832


namespace minimize_triangle_side_l988_98808

noncomputable def minimizeTriangleSide (t : ℝ) (C : ℝ) : ℝ × ℝ × ℝ :=
  let a := (2 * t / Real.sin C) ^ (1/2)
  let b := a
  let c := 2 * (t * Real.tan (C/2)) ^ (1/2)
  (a, b, c)

theorem minimize_triangle_side (t : ℝ) (C : ℝ) (h1 : t > 0) (h2 : 0 < C ∧ C < π) :
  let (a, b, c) := minimizeTriangleSide t C
  (∀ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 2 * t = a' * b' * Real.sin C →
    c ≤ (a'^2 + b'^2 - 2*a'*b'*Real.cos C)^(1/2)) ∧
  a = b ∧
  c = 2 * (t * Real.tan (C/2))^(1/2) :=
by sorry

end minimize_triangle_side_l988_98808


namespace arithmetic_square_root_of_16_l988_98875

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end arithmetic_square_root_of_16_l988_98875


namespace function_properties_imply_cosine_and_value_l988_98831

/-- The function f(x) = sin(ωx + φ) with given properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ Real.sin (ω * x + φ)

/-- The theorem statement -/
theorem function_properties_imply_cosine_and_value
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_distance : ∃ (x₁ x₂ : ℝ), abs (x₁ - x₂) = π ∧ abs (f ω φ x₁ - f ω φ x₂) = 2)
  (α : ℝ)
  (h_sum : Real.sin α + f ω φ α = 2/3) :
  (∀ x, f ω φ x = Real.cos x) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = -5/9) :=
by sorry

end function_properties_imply_cosine_and_value_l988_98831


namespace school_trip_photos_l988_98816

theorem school_trip_photos (c : ℕ) : 
  (3 * c = c + 12) →  -- Lisa and Robert have the same number of photos
  c = 6               -- Claire took 6 photos
  := by sorry

end school_trip_photos_l988_98816


namespace solution_set_f_greater_than_two_range_of_k_l988_98861

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -2} := by sorry

-- Theorem for the second part of the problem
theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3) (-1), f x ≤ k * x + 1) ↔ k ≤ -1 := by sorry

end solution_set_f_greater_than_two_range_of_k_l988_98861


namespace equal_parts_in_one_to_one_mix_l988_98864

/-- Represents a substrate composition with parts of bark, peat, and sand -/
structure Substrate :=
  (bark : ℚ)
  (peat : ℚ)
  (sand : ℚ)

/-- Orchid-1 substrate composition -/
def orchid1 : Substrate :=
  { bark := 3
    peat := 2
    sand := 1 }

/-- Orchid-2 substrate composition -/
def orchid2 : Substrate :=
  { bark := 1
    peat := 2
    sand := 3 }

/-- Mixes two substrates in given proportions -/
def mixSubstrates (s1 s2 : Substrate) (r1 r2 : ℚ) : Substrate :=
  { bark := r1 * s1.bark + r2 * s2.bark
    peat := r1 * s1.peat + r2 * s2.peat
    sand := r1 * s1.sand + r2 * s2.sand }

/-- Checks if all components of a substrate are equal -/
def hasEqualParts (s : Substrate) : Prop :=
  s.bark = s.peat ∧ s.peat = s.sand

theorem equal_parts_in_one_to_one_mix :
  hasEqualParts (mixSubstrates orchid1 orchid2 1 1) :=
by sorry


end equal_parts_in_one_to_one_mix_l988_98864


namespace quadratic_inequality_l988_98819

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 18 < 0 ↔ -3 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l988_98819


namespace right_triangle_segment_ratio_l988_98843

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  r * s = c^2 →      -- Geometric mean theorem
  r * c = a^2 →      -- Geometric mean theorem
  s * c = b^2 →      -- Geometric mean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end right_triangle_segment_ratio_l988_98843


namespace polynomial_equation_solution_l988_98834

-- Define the set of real polynomials
def RealPolynomial := Polynomial ℝ

-- Define the condition for a, b, c
def SumProductZero (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the equation that P must satisfy
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SumProductZero a b c →
    P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)

-- Define the form of the solution polynomial
def IsSolutionForm (P : RealPolynomial) : Prop :=
  ∃ u v : ℝ, P = Polynomial.monomial 4 u + Polynomial.monomial 2 v

-- State the theorem
theorem polynomial_equation_solution :
  ∀ P : RealPolynomial, SatisfiesEquation P → IsSolutionForm P :=
sorry

end polynomial_equation_solution_l988_98834


namespace impossible_equal_sums_l988_98873

/-- A configuration of numbers on a triangle with medians -/
structure TriangleConfig where
  vertices : Fin 3 → ℕ
  midpoints : Fin 3 → ℕ
  center : ℕ

/-- The sum of numbers on a side of the triangle -/
def side_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints i + config.vertices ((i + 1) % 3)

/-- The sum of numbers on a median of the triangle -/
def median_sum (config : TriangleConfig) (i : Fin 3) : ℕ :=
  config.vertices i + config.midpoints ((i + 1) % 3) + config.center

/-- Predicate to check if a configuration is valid -/
def is_valid_config (config : TriangleConfig) : Prop :=
  (∀ i : Fin 3, config.vertices i ≤ 7) ∧
  (∀ i : Fin 3, config.midpoints i ≤ 7) ∧
  (config.center ≤ 7) ∧
  (config.vertices 0 + config.vertices 1 + config.vertices 2 +
   config.midpoints 0 + config.midpoints 1 + config.midpoints 2 +
   config.center = 28)

/-- Predicate to check if a configuration has equal sums -/
def has_equal_sums (config : TriangleConfig) : Prop :=
  ∃ x : ℕ, (∀ i : Fin 3, side_sum config i = x) ∧
            (∀ i : Fin 3, median_sum config i = x)

theorem impossible_equal_sums : ¬∃ config : TriangleConfig, 
  is_valid_config config ∧ has_equal_sums config := by
  sorry

end impossible_equal_sums_l988_98873


namespace ac_length_l988_98886

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length conditions
  ab_length : dist A B = 7
  bc_length : dist B C = 24
  -- Area condition
  area : abs ((A.1 - C.1) * (B.2 - A.2) - (A.2 - C.2) * (B.1 - A.1)) / 2 = 84
  -- Median condition
  median_length : dist A ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = 12.5

/-- Theorem about the length of AC in the special triangle -/
theorem ac_length (t : SpecialTriangle) : dist t.A t.C = 25 := by
  sorry

end ac_length_l988_98886


namespace symmetry_coordinates_l988_98897

/-- Two points are symmetrical about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are the same. -/
def symmetrical_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_coordinates :
  let p : ℝ × ℝ := (4, -5)
  let q : ℝ × ℝ := (a, b)
  symmetrical_about_y_axis p q → a = -4 ∧ b = -5 := by
sorry

end symmetry_coordinates_l988_98897


namespace opposite_of_2023_l988_98857

theorem opposite_of_2023 : 
  ∃ y : ℤ, y + 2023 = 0 ∧ y = -2023 := by
  sorry

end opposite_of_2023_l988_98857


namespace relay_race_arrangements_l988_98804

def number_of_students : ℕ := 4
def fixed_position : ℕ := 1
def available_positions : ℕ := number_of_students - fixed_position

theorem relay_race_arrangements :
  (available_positions.factorial) = 6 := by
  sorry

end relay_race_arrangements_l988_98804


namespace B_power_15_minus_3_power_14_l988_98865

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 3]

theorem B_power_15_minus_3_power_14 : 
  B^15 - 3 • B^14 = !![4^14, 4^14; 0, 0] := by sorry

end B_power_15_minus_3_power_14_l988_98865


namespace impossible_tiling_l988_98829

/-- Represents a tile type -/
inductive TileType
| TwoByTwo
| OneByFour

/-- Represents a set of tiles -/
structure TileSet where
  twoByTwo : Nat
  oneByFour : Nat

/-- Represents a rectangular box -/
structure Box where
  length : Nat
  width : Nat

/-- Checks if a box can be tiled with a given tile set -/
def canTile (box : Box) (tiles : TileSet) : Prop :=
  sorry

/-- The main theorem -/
theorem impossible_tiling (box : Box) (initialTiles : TileSet) :
  canTile box initialTiles →
  ¬canTile box { twoByTwo := initialTiles.twoByTwo - 1, oneByFour := initialTiles.oneByFour + 1 } :=
sorry

end impossible_tiling_l988_98829


namespace repeating_decimal_equals_fraction_l988_98853

/-- The repeating decimal 0.215215215... -/
def repeating_decimal : ℚ := 0.215215215

/-- The fraction 215/999 -/
def fraction : ℚ := 215 / 999

/-- Theorem stating that the repeating decimal 0.215215215... is equal to the fraction 215/999 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l988_98853


namespace wizard_concoction_combinations_l988_98851

/-- Represents the number of herbs available --/
def num_herbs : ℕ := 4

/-- Represents the number of crystals available --/
def num_crystals : ℕ := 6

/-- Represents the number of incompatible combinations --/
def num_incompatible : ℕ := 3

/-- Theorem stating the number of valid combinations for the wizard's concoction --/
theorem wizard_concoction_combinations : 
  num_herbs * num_crystals - num_incompatible = 21 := by
  sorry

end wizard_concoction_combinations_l988_98851


namespace matilda_has_420_jellybeans_l988_98881

/-- The number of jellybeans Steve has -/
def steve_jellybeans : ℕ := 84

/-- The number of jellybeans Matt has -/
def matt_jellybeans : ℕ := 10 * steve_jellybeans

/-- The number of jellybeans Matilda has -/
def matilda_jellybeans : ℕ := matt_jellybeans / 2

/-- Theorem stating that Matilda has 420 jellybeans -/
theorem matilda_has_420_jellybeans : matilda_jellybeans = 420 := by
  sorry

end matilda_has_420_jellybeans_l988_98881


namespace inscribed_rectangle_height_l988_98885

/-- 
Given a triangle with base b and height h, and a rectangle inscribed in it such that:
1. The base of the rectangle coincides with the base of the triangle
2. The height of the rectangle is half its base
Prove that the height of the rectangle x is equal to bh / (2h + b)
-/
theorem inscribed_rectangle_height (b h : ℝ) (h1 : 0 < b) (h2 : 0 < h) : 
  ∃ x : ℝ, x > 0 ∧ x = b * h / (2 * h + b) := by
  sorry

end inscribed_rectangle_height_l988_98885


namespace association_and_likelihood_ratio_l988_98866

-- Define the contingency table
def excellent_math_excellent_chinese : ℕ := 45
def excellent_math_not_excellent_chinese : ℕ := 35
def not_excellent_math_excellent_chinese : ℕ := 45
def not_excellent_math_not_excellent_chinese : ℕ := 75

def total_sample_size : ℕ := 200

-- Define the chi-square test statistic
def chi_square_statistic : ℚ :=
  (total_sample_size * (excellent_math_excellent_chinese * not_excellent_math_not_excellent_chinese - 
  excellent_math_not_excellent_chinese * not_excellent_math_excellent_chinese)^2) / 
  ((excellent_math_excellent_chinese + excellent_math_not_excellent_chinese) * 
  (not_excellent_math_excellent_chinese + not_excellent_math_not_excellent_chinese) * 
  (excellent_math_excellent_chinese + not_excellent_math_excellent_chinese) * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

-- Define the critical value at α = 0.01
def critical_value : ℚ := 6635 / 1000

-- Define the likelihood ratio L(B|A)
def likelihood_ratio : ℚ := 
  (not_excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese)) / 
  (excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

theorem association_and_likelihood_ratio : 
  chi_square_statistic > critical_value ∧ likelihood_ratio = 15 / 7 := by sorry

end association_and_likelihood_ratio_l988_98866


namespace area_of_rectangle_S_l988_98863

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of shapes within the larger square -/
structure Configuration where
  largerSquare : Square
  rectangle : Rectangle
  smallerSquare : Square
  rectangleS : Rectangle

/-- The conditions of the problem -/
def validConfiguration (c : Configuration) : Prop :=
  c.rectangle.width = 2 ∧
  c.rectangle.height = 4 ∧
  c.smallerSquare.side = 2 ∧
  c.largerSquare.side ≥ 4 ∧
  c.largerSquare.side ^ 2 = 
    c.rectangle.width * c.rectangle.height +
    c.smallerSquare.side ^ 2 +
    c.rectangleS.width * c.rectangleS.height

theorem area_of_rectangle_S (c : Configuration) 
  (h : validConfiguration c) : 
  c.rectangleS.width * c.rectangleS.height = 4 :=
sorry

end area_of_rectangle_S_l988_98863


namespace multiply_mistake_l988_98840

theorem multiply_mistake (x : ℝ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end multiply_mistake_l988_98840


namespace fractional_equation_root_l988_98811

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 2) - 2 = m / (x - 2)) → m = 2 := by
  sorry

end fractional_equation_root_l988_98811


namespace sons_age_l988_98856

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 10 := by
sorry

end sons_age_l988_98856


namespace power_equality_l988_98850

theorem power_equality (n : ℕ) : 9^4 = 3^n → n = 8 := by
  sorry

end power_equality_l988_98850


namespace binomial_identity_sum_identity_l988_98820

def binomial (n p : ℕ) : ℕ := if p ≤ n then n.factorial / (p.factorial * (n - p).factorial) else 0

theorem binomial_identity (n p : ℕ) (h : n ≥ p ∧ p ≥ 1) :
  binomial n p = (Finset.range (n - p + 1)).sum (fun i => binomial (n - 1 - i) (p - 1)) :=
sorry

theorem sum_identity :
  (Finset.range 97).sum (fun k => (k + 1) * (k + 2) * (k + 3)) = 23527350 :=
sorry

end binomial_identity_sum_identity_l988_98820


namespace quadratic_equation_set_equivalence_l988_98842

theorem quadratic_equation_set_equivalence :
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end quadratic_equation_set_equivalence_l988_98842


namespace distance_from_origin_to_point_l988_98871

theorem distance_from_origin_to_point (z : ℂ) : 
  z = 1260 + 1680 * Complex.I → Complex.abs z = 2100 := by
  sorry

end distance_from_origin_to_point_l988_98871


namespace ted_age_l988_98896

/-- Given that Ted's age is 20 years less than three times Sally's age,
    and the sum of their ages is 70, prove that Ted is 47.5 years old. -/
theorem ted_age (sally_age : ℝ) (ted_age : ℝ) 
  (h1 : ted_age = 3 * sally_age - 20)
  (h2 : ted_age + sally_age = 70) : 
  ted_age = 47.5 := by
  sorry

end ted_age_l988_98896


namespace number_difference_proof_l988_98859

theorem number_difference_proof (s l : ℕ) : 
  (∃ x : ℕ, l = 2 * s - x) →  -- One number is some less than twice another
  s + l = 39 →               -- Their sum is 39
  s = 14 →                   -- The smaller number is 14
  2 * s - l = 3 :=           -- The difference between twice the smaller number and the larger number is 3
by
  sorry

end number_difference_proof_l988_98859


namespace quadratic_function_range_l988_98800

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₀ : ℝ, |x₀^2 + a*x₀ + 1| ≤ 1/4 ∧ |(x₀+1)^2 + a*(x₀+1) + 1| ≤ 1/4) → 
  a ∈ Set.Icc (-Real.sqrt 6) (-2) ∪ Set.Icc 2 (Real.sqrt 6) :=
by sorry

end quadratic_function_range_l988_98800


namespace opposite_face_is_B_l988_98818

-- Define the faces of the cube
inductive Face : Type
| X | A | B | C | D | E

-- Define the net structure
structure Net :=
  (faces : Finset Face)
  (center : Face)
  (surrounding : List Face)
  (adjacent_to_A : Face)
  (adjacent_to_D : Face)

-- Define the property of being opposite in a cube
def is_opposite (f1 f2 : Face) : Prop := sorry

-- Define the cube folding function
def fold_to_cube (n : Net) : Prop := sorry

-- Theorem statement
theorem opposite_face_is_B (n : Net) : 
  n.faces.card = 6 ∧ 
  n.center = Face.X ∧ 
  n.surrounding = [Face.A, Face.B, Face.D] ∧
  n.adjacent_to_A = Face.C ∧
  n.adjacent_to_D = Face.E ∧
  fold_to_cube n →
  is_opposite Face.X Face.B :=
sorry

end opposite_face_is_B_l988_98818


namespace complement_M_intersect_N_l988_98846

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 3}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | x < -3} := by sorry

end complement_M_intersect_N_l988_98846


namespace gwen_total_books_l988_98872

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 32 := by sorry

end gwen_total_books_l988_98872


namespace dress_discount_percentage_l988_98889

/-- Calculates the final discount percentage for a dress purchase with multiple discounts -/
theorem dress_discount_percentage (original_price : ℝ) (store_discount : ℝ) (member_discount : ℝ) :
  original_price = 350 →
  store_discount = 0.20 →
  member_discount = 0.10 →
  let price_after_store_discount := original_price * (1 - store_discount)
  let final_price := price_after_store_discount * (1 - member_discount)
  let total_discount := original_price - final_price
  let final_discount_percentage := (total_discount / original_price) * 100
  ∃ ε > 0, |final_discount_percentage - 28| < ε :=
by
  sorry


end dress_discount_percentage_l988_98889


namespace mean_proportional_existence_l988_98854

theorem mean_proportional_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x ^ 2 = a * b :=
by sorry

end mean_proportional_existence_l988_98854


namespace max_of_min_is_sqrt_two_l988_98838

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 :=
by sorry

end max_of_min_is_sqrt_two_l988_98838


namespace circle_area_difference_l988_98880

theorem circle_area_difference (r₁ r₂ r : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  π * r₂^2 - π * r₁^2 = π * r^2 → r = 20 :=
by sorry

end circle_area_difference_l988_98880


namespace hexagon_enclosed_by_polygons_l988_98824

/-- A regular hexagon is enclosed by m regular n-sided polygons, where three polygons meet at each vertex of the hexagon. -/
theorem hexagon_enclosed_by_polygons (m : ℕ) (n : ℕ) : n = 18 := by
  sorry

end hexagon_enclosed_by_polygons_l988_98824


namespace two_word_sentences_count_correct_count_l988_98862

def word : String := "YARIŞMA"

theorem two_word_sentences_count : ℕ :=
  let n : ℕ := word.length
  let repeated_letter_count : ℕ := 2  -- 'A' appears twice
  let permutations : ℕ := n.factorial / repeated_letter_count.factorial
  let space_positions : ℕ := n + 1
  permutations * space_positions

theorem correct_count : two_word_sentences_count = 20160 := by
  sorry

end two_word_sentences_count_correct_count_l988_98862


namespace regular_tetrahedron_edges_l988_98801

/-- A regular tetrahedron is a tetrahedron in which all faces are congruent equilateral triangles. -/
def RegularTetrahedron : Type := sorry

/-- The number of edges in a geometric shape. -/
def num_edges (shape : Type) : ℕ := sorry

/-- Theorem: A regular tetrahedron has 6 edges. -/
theorem regular_tetrahedron_edges : num_edges RegularTetrahedron = 6 := by
  sorry

end regular_tetrahedron_edges_l988_98801


namespace decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l988_98882

theorem decimal_digit_of_fraction (n : ℕ) (a b : ℕ) (h : b ≠ 0) :
  ∃ (d : ℕ), d < 10 ∧ d = (a * 10^n) % b :=
sorry

theorem thirteenth_over_seventeen_150th_digit :
  ∃ (d : ℕ), d < 10 ∧ d = (13 * 10^150) % 17 ∧ d = 1 :=
sorry

end decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l988_98882


namespace sum_of_reciprocals_l988_98826

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 := by
sorry

end sum_of_reciprocals_l988_98826


namespace megan_markers_count_l988_98894

/-- The number of markers Megan has after receiving and giving away some -/
def final_markers (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

/-- Theorem stating that Megan's final number of markers is correct -/
theorem megan_markers_count :
  final_markers 217 109 35 = 291 :=
by sorry

end megan_markers_count_l988_98894


namespace max_median_soda_sales_l988_98810

/-- Represents the soda sales data for a weekend -/
structure SodaSales where
  totalCans : ℕ
  totalCustomers : ℕ
  minCansPerCustomer : ℕ

/-- Calculates the maximum possible median number of cans bought per customer -/
def maxPossibleMedian (sales : SodaSales) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem max_median_soda_sales (sales : SodaSales)
  (h1 : sales.totalCans = 300)
  (h2 : sales.totalCustomers = 120)
  (h3 : sales.minCansPerCustomer = 2) :
  maxPossibleMedian sales = 3 :=
  sorry

end max_median_soda_sales_l988_98810


namespace inequality_solution_set_l988_98827

theorem inequality_solution_set (x : ℝ) : (3 - 2*x) * (x + 1) ≤ 0 ↔ x < -1 ∨ x ≥ 3/2 := by
  sorry

end inequality_solution_set_l988_98827


namespace max_pieces_in_5x5_grid_l988_98887

theorem max_pieces_in_5x5_grid : ∀ (n : ℕ),
  (∃ (areas : List ℕ), 
    areas.length = n ∧ 
    areas.sum = 25 ∧ 
    areas.Nodup ∧ 
    (∀ a ∈ areas, a > 0)) →
  n ≤ 6 :=
by sorry

end max_pieces_in_5x5_grid_l988_98887


namespace sqrt_equation_solution_l988_98868

theorem sqrt_equation_solution : 
  let x : ℝ := 12/5
  (Real.sqrt (6*x)) / (Real.sqrt (4*(x-2))) = 3 := by
  sorry

end sqrt_equation_solution_l988_98868


namespace complementary_angles_can_be_equal_l988_98891

-- Define what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- State the theorem
theorem complementary_angles_can_be_equal :
  ∃ (α : ℝ), complementary α α :=
sorry

-- The existence of such an angle pair disproves the statement
-- "Two complementary angles are not equal"

end complementary_angles_can_be_equal_l988_98891


namespace place_face_value_difference_l988_98883

def number : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  if n / 100 % 10 = d then d * 10 else 0

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value number digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end place_face_value_difference_l988_98883


namespace repeating_ones_not_square_l988_98833

/-- Defines a function that returns a number consisting of n repeating 1's -/
def repeatingOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that for any positive natural number n, 
    the number consisting of n repeating 1's is not a perfect square -/
theorem repeating_ones_not_square (n : ℕ) (h : n > 0) : 
  ¬ ∃ m : ℕ, (repeatingOnes n) = m^2 := by
  sorry

end repeating_ones_not_square_l988_98833


namespace average_shift_l988_98884

theorem average_shift (a b c : ℝ) : 
  (a + b + c) / 3 = 5 → ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 := by
  sorry

end average_shift_l988_98884


namespace stratified_sampling_medium_supermarkets_l988_98821

theorem stratified_sampling_medium_supermarkets 
  (total_sample : ℕ) 
  (large_supermarkets : ℕ) 
  (medium_supermarkets : ℕ) 
  (small_supermarkets : ℕ) 
  (h_total_sample : total_sample = 100)
  (h_large : large_supermarkets = 200)
  (h_medium : medium_supermarkets = 400)
  (h_small : small_supermarkets = 1400) : 
  (total_sample * medium_supermarkets) / (large_supermarkets + medium_supermarkets + small_supermarkets) = 20 := by
sorry

end stratified_sampling_medium_supermarkets_l988_98821


namespace log_1458_between_consecutive_integers_l988_98805

theorem log_1458_between_consecutive_integers (c d : ℤ) : 
  (c : ℝ) < Real.log 1458 / Real.log 10 ∧ 
  Real.log 1458 / Real.log 10 < (d : ℝ) ∧ 
  d = c + 1 → 
  c + d = 7 := by
sorry

end log_1458_between_consecutive_integers_l988_98805


namespace line_circle_intersection_l988_98852

/-- The intersection points of a line and a circle -/
theorem line_circle_intersection :
  let line := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 9 }
  let point1 := ((1 + Real.sqrt 17) / 2, (1 - Real.sqrt 17) / 2)
  let point2 := ((1 - Real.sqrt 17) / 2, (1 + Real.sqrt 17) / 2)
  (point1 ∈ line ∧ point1 ∈ circle) ∧ 
  (point2 ∈ line ∧ point2 ∈ circle) ∧
  (∀ p ∈ line ∩ circle, p = point1 ∨ p = point2) :=
by
  sorry


end line_circle_intersection_l988_98852


namespace crackers_distribution_l988_98844

/-- The number of crackers Matthew had initially -/
def total_crackers : ℕ := 32

/-- The number of friends Matthew gave crackers to -/
def num_friends : ℕ := 4

/-- The number of crackers each friend received -/
def crackers_per_friend : ℕ := total_crackers / num_friends

theorem crackers_distribution :
  crackers_per_friend = 8 := by sorry

end crackers_distribution_l988_98844


namespace partnership_problem_l988_98828

/-- Partnership problem -/
theorem partnership_problem (a_months b_months : ℕ) (b_contribution total_profit a_share : ℝ) 
  (h1 : a_months = 8)
  (h2 : b_months = 5)
  (h3 : b_contribution = 6000)
  (h4 : total_profit = 8400)
  (h5 : a_share = 4800) :
  ∃ (a_contribution : ℝ),
    a_contribution * a_months * (total_profit - a_share) = 
    b_contribution * b_months * a_share ∧ 
    a_contribution = 5000 := by
  sorry

end partnership_problem_l988_98828
