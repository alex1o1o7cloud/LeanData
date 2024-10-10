import Mathlib

namespace product_of_roots_l106_10625

theorem product_of_roots (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 1 = 0) → (x₂^2 + x₂ - 1 = 0) → x₁ * x₂ = -1 := by
  sorry

end product_of_roots_l106_10625


namespace power_function_symmetry_l106_10688

/-- A function f is a power function if it can be written as f(x) = kx^n for some constant k and real number n. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, f x = k * x ^ n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x in the domain of f. -/
def isSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem stating the properties of the function f(x) = (2m^2 - m)x^(2m+3) -/
theorem power_function_symmetry (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (2 * m^2 - m) * x^(2*m + 3)
  isPowerFunction f ∧ isSymmetricAboutYAxis f →
  (m = -1/2) ∧
  (∀ a : ℝ, 3/2 < a ∧ a < 2 ↔ (a - 1)^m < (2*a - 3)^m) :=
by sorry

end power_function_symmetry_l106_10688


namespace unique_abc_solution_l106_10666

theorem unique_abc_solution (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end unique_abc_solution_l106_10666


namespace inequality_proof_l106_10615

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end inequality_proof_l106_10615


namespace quadratic_equation_problem_l106_10613

theorem quadratic_equation_problem (k : ℝ) : 
  (∀ x, 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 → 
    (6 * Real.sqrt 3)^2 - 4 * 4 * k = 18) → 
  k = 45/8 ∧ ∃ x y, x ≠ y ∧ 4 * x^2 - 6 * x * Real.sqrt 3 + k = 0 ∧ 
                           4 * y^2 - 6 * y * Real.sqrt 3 + k = 0 :=
by sorry

end quadratic_equation_problem_l106_10613


namespace triangle_properties_l106_10628

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * Real.sin (t.A - t.C) = Real.sin t.B ∧
  t.AB = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 3 * Real.sqrt 10 / 10 ∧
  ∃ (height : Real), height = 6 ∧ 
    height * t.AB / 2 = Real.sin t.C * (t.AB * Real.sin t.B / Real.sin t.C) * (t.AB * Real.sin t.A / Real.sin t.C) / 2 :=
by sorry

end triangle_properties_l106_10628


namespace french_books_count_l106_10616

/-- The number of English books -/
def num_english_books : ℕ := 11

/-- The total number of arrangement ways -/
def total_arrangements : ℕ := 220

/-- The number of French books -/
def num_french_books : ℕ := 3

/-- The number of slots for French books -/
def num_slots : ℕ := num_english_books + 1

theorem french_books_count :
  (Nat.choose num_slots num_french_books = total_arrangements) ∧
  (∀ k : ℕ, k ≠ num_french_books → Nat.choose num_slots k ≠ total_arrangements) :=
sorry

end french_books_count_l106_10616


namespace p_neither_sufficient_nor_necessary_for_q_l106_10608

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (a b : ℝ) : Prop := a * b = -1

/-- The condition p: ax + y + 1 = 0 is perpendicular to ax - y + 2 = 0 -/
def p (a : ℝ) : Prop := perpendicular a (-a)

/-- The condition q: a = 1 -/
def q : ℝ → Prop := (· = 1)

/-- p is neither sufficient nor necessary for q -/
theorem p_neither_sufficient_nor_necessary_for_q :
  (¬∀ a, p a → q a) ∧ (¬∀ a, q a → p a) := by sorry

end p_neither_sufficient_nor_necessary_for_q_l106_10608


namespace smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l106_10622

theorem smallest_four_digit_perfect_square_multiple :
  ∀ m : ℕ, m ≥ 1000 → m < 1029 → ¬∃ n : ℕ, 21 * m = n * n :=
by
  sorry

theorem smallest_four_digit_perfect_square_multiple_exists :
  ∃ n : ℕ, 21 * 1029 = n * n :=
by
  sorry

#check smallest_four_digit_perfect_square_multiple
#check smallest_four_digit_perfect_square_multiple_exists

end smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l106_10622


namespace island_width_is_five_l106_10676

/-- Represents a rectangular island -/
structure Island where
  length : ℝ
  width : ℝ
  area : ℝ

/-- The area of a rectangular island is equal to its length multiplied by its width -/
axiom island_area (i : Island) : i.area = i.length * i.width

/-- Given an island with area 50 square miles and length 10 miles, prove its width is 5 miles -/
theorem island_width_is_five (i : Island) 
  (h_area : i.area = 50) 
  (h_length : i.length = 10) : 
  i.width = 5 := by
sorry

end island_width_is_five_l106_10676


namespace kolakoski_next_eight_terms_l106_10695

/-- The Kolakoski sequence -/
def kolakoski : ℕ → Fin 2
  | 0 => 0  -- represents 1
  | 1 => 1  -- represents 2
  | 2 => 1  -- represents 2
  | n + 3 => sorry

/-- The run-length encoding of the Kolakoski sequence -/
def kolakoski_rle : ℕ → Fin 2
  | n => kolakoski n

theorem kolakoski_next_eight_terms :
  (List.range 8).map (fun i => kolakoski (i + 12)) = [1, 1, 0, 0, 1, 0, 0, 1] := by
  sorry

#check kolakoski_next_eight_terms

end kolakoski_next_eight_terms_l106_10695


namespace stevens_falls_l106_10669

theorem stevens_falls (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) 
  (h1 : stephanie_falls = steven_falls + 13)
  (h2 : sonya_falls = 6)
  (h3 : sonya_falls = stephanie_falls / 2 - 2) : 
  steven_falls = 3 := by
  sorry

end stevens_falls_l106_10669


namespace remaining_students_average_l106_10671

theorem remaining_students_average (total_students : ℕ) (class_average : ℚ)
  (group1_fraction : ℚ) (group1_average : ℚ)
  (group2_fraction : ℚ) (group2_average : ℚ)
  (group3_fraction : ℚ) (group3_average : ℚ)
  (h1 : total_students = 120)
  (h2 : class_average = 84)
  (h3 : group1_fraction = 1/4)
  (h4 : group1_average = 96)
  (h5 : group2_fraction = 1/5)
  (h6 : group2_average = 75)
  (h7 : group3_fraction = 1/8)
  (h8 : group3_average = 90) :
  let remaining_students := total_students - (group1_fraction * total_students + group2_fraction * total_students + group3_fraction * total_students)
  let remaining_average := (total_students * class_average - (group1_fraction * total_students * group1_average + group2_fraction * total_students * group2_average + group3_fraction * total_students * group3_average)) / remaining_students
  remaining_average = 4050 / 51 := by
  sorry

end remaining_students_average_l106_10671


namespace smallest_x_is_correct_smallest_x_works_l106_10656

/-- The smallest positive integer x such that 1800x is a perfect cube -/
def smallest_x : ℕ := 15

theorem smallest_x_is_correct :
  ∀ y : ℕ, y > 0 → (∃ m : ℕ, 1800 * y = m^3) → y ≥ smallest_x :=
by sorry

theorem smallest_x_works :
  ∃ m : ℕ, 1800 * smallest_x = m^3 :=
by sorry

end smallest_x_is_correct_smallest_x_works_l106_10656


namespace sine_inequality_range_l106_10693

theorem sine_inequality_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin x < a) → a > -1 := by
  sorry

end sine_inequality_range_l106_10693


namespace initial_money_calculation_l106_10684

/-- Calculates the initial amount of money given spending habits and remaining balance --/
theorem initial_money_calculation 
  (spend_per_trip : ℕ)
  (trips_per_month : ℕ)
  (months : ℕ)
  (money_left : ℕ)
  (h1 : spend_per_trip = 2)
  (h2 : trips_per_month = 4)
  (h3 : months = 12)
  (h4 : money_left = 104) :
  spend_per_trip * trips_per_month * months + money_left = 200 := by
  sorry

#check initial_money_calculation

end initial_money_calculation_l106_10684


namespace star_properties_l106_10603

-- Define the star operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_properties :
  -- There exists an identity element E
  ∃ E : ℝ, (∀ a : ℝ, star a E = a) ∧ (star E E = E) ∧
  -- The operation is commutative
  (∀ a b : ℝ, star a b = star b a) ∧
  -- The operation is associative
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) :=
sorry

end star_properties_l106_10603


namespace min_chess_pieces_chess_pieces_solution_l106_10662

theorem min_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → n ≥ 103 :=
by sorry

theorem chess_pieces_solution : 
  ∃ (n : ℕ), (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) ∧ n = 103 :=
by sorry

end min_chess_pieces_chess_pieces_solution_l106_10662


namespace integer_solutions_of_equation_l106_10629

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x + y + x * y = 500 ↔ 
    ((x = 0 ∧ y = 500) ∨ 
     (x = -2 ∧ y = -502) ∨ 
     (x = 2 ∧ y = 166) ∨ 
     (x = -4 ∧ y = -168)) := by
  sorry

end integer_solutions_of_equation_l106_10629


namespace train_speed_problem_l106_10609

theorem train_speed_problem (distance : ℝ) (time : ℝ) (speed_A : ℝ) : 
  distance = 480 →
  time = 2.5 →
  speed_A = 102 →
  (distance / time) - speed_A = 90 := by
sorry

end train_speed_problem_l106_10609


namespace complement_of_A_union_B_in_U_l106_10614

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = {5} := by sorry

end complement_of_A_union_B_in_U_l106_10614


namespace fraction_addition_l106_10612

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end fraction_addition_l106_10612


namespace interior_angles_sum_l106_10652

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles a quadrilateral can be divided into -/
def quadrilateral_triangles : ℕ := 2

/-- The number of triangles a pentagon can be divided into -/
def pentagon_triangles : ℕ := 3

/-- The number of triangles a convex n-gon can be divided into -/
def n_gon_triangles (n : ℕ) : ℕ := n - 2

/-- The sum of interior angles of a quadrilateral -/
def quadrilateral_angle_sum : ℝ := triangle_angle_sum * quadrilateral_triangles

/-- The sum of interior angles of a convex pentagon -/
def pentagon_angle_sum : ℝ := triangle_angle_sum * pentagon_triangles

/-- The sum of interior angles of a convex n-gon -/
def n_gon_angle_sum (n : ℕ) : ℝ := triangle_angle_sum * n_gon_triangles n

theorem interior_angles_sum :
  (quadrilateral_angle_sum = 360) ∧
  (pentagon_angle_sum = 540) ∧
  (∀ n : ℕ, n_gon_angle_sum n = 180 * (n - 2)) :=
sorry

end interior_angles_sum_l106_10652


namespace lowest_possible_score_l106_10627

def exam_count : ℕ := 4
def max_score : ℕ := 100
def first_exam_score : ℕ := 84
def second_exam_score : ℕ := 67
def target_average : ℕ := 75

theorem lowest_possible_score :
  ∃ (third_exam_score fourth_exam_score : ℕ),
    third_exam_score ≤ max_score ∧
    fourth_exam_score ≤ max_score ∧
    (first_exam_score + second_exam_score + third_exam_score + fourth_exam_score) / exam_count ≥ target_average ∧
    (third_exam_score = 49 ∨ fourth_exam_score = 49) ∧
    ∀ (x y : ℕ),
      x ≤ max_score →
      y ≤ max_score →
      (first_exam_score + second_exam_score + x + y) / exam_count ≥ target_average →
      x ≥ 49 ∧ y ≥ 49 :=
by sorry

end lowest_possible_score_l106_10627


namespace greatest_common_multiple_10_15_under_120_l106_10657

theorem greatest_common_multiple_10_15_under_120 : 
  ∃ (n : ℕ), n = Nat.lcm 10 15 ∧ n < 120 ∧ ∀ m : ℕ, (m = Nat.lcm 10 15 ∧ m < 120) → m ≤ n :=
by sorry

end greatest_common_multiple_10_15_under_120_l106_10657


namespace first_non_divisor_is_seven_l106_10620

def is_valid_integer (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0

theorem first_non_divisor_is_seven :
  ∃ (S : Finset ℕ), 
    Finset.card S = 26 ∧ 
    (∀ n ∈ S, is_valid_integer n) ∧
    (∀ k > 5, k < 7 → ∃ n ∈ S, n % k = 0) ∧
    (∀ n ∈ S, n % 7 ≠ 0) :=
sorry

end first_non_divisor_is_seven_l106_10620


namespace cosine_min_phase_l106_10647

/-- Given a cosine function y = a cos(bx + c) where a, b, and c are positive constants,
    if the function reaches its first minimum at x = π/(2b), then c = π/2. -/
theorem cosine_min_phase (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, x ≥ 0 → a * Real.cos (b * x + c) ≥ a * Real.cos (b * (Real.pi / (2 * b)) + c)) →
  c = Real.pi / 2 := by
  sorry

end cosine_min_phase_l106_10647


namespace sum_remainder_mod_nine_l106_10637

theorem sum_remainder_mod_nine (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end sum_remainder_mod_nine_l106_10637


namespace inequality_proof_l106_10661

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end inequality_proof_l106_10661


namespace peters_pumpkin_profit_l106_10655

/-- Represents the total amount of money collected from selling pumpkins -/
def total_money (jumbo_price regular_price : ℝ) (total_pumpkins regular_pumpkins : ℕ) : ℝ :=
  regular_price * regular_pumpkins + jumbo_price * (total_pumpkins - regular_pumpkins)

/-- Theorem stating that Peter's total money collected is $395.00 -/
theorem peters_pumpkin_profit :
  total_money 9 4 80 65 = 395 := by
  sorry

end peters_pumpkin_profit_l106_10655


namespace angelina_driving_equation_l106_10631

/-- Represents the driving scenario of Angelina --/
structure DrivingScenario where
  initial_speed : ℝ
  rest_time : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for Angelina's driving time before rest --/
def driving_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.initial_speed * t + s.final_speed * (s.total_time - s.rest_time / 60 - t) = s.total_distance

/-- Theorem stating that the given equation correctly represents Angelina's driving scenario --/
theorem angelina_driving_equation :
  ∃ (s : DrivingScenario),
    s.initial_speed = 60 ∧
    s.rest_time = 15 ∧
    s.final_speed = 90 ∧
    s.total_distance = 255 ∧
    s.total_time = 4 ∧
    ∀ (t : ℝ), driving_equation s t ↔ (60 * t + 90 * (15 / 4 - t) = 255) :=
  sorry

end angelina_driving_equation_l106_10631


namespace vacation_cost_equality_l106_10646

/-- Proves that t - d + s = 20 given the vacation cost conditions --/
theorem vacation_cost_equality (tom_paid dorothy_paid sammy_paid t d s : ℚ) :
  tom_paid = 150 →
  dorothy_paid = 160 →
  sammy_paid = 210 →
  let total := tom_paid + dorothy_paid + sammy_paid
  let per_person := total / 3
  t = per_person - tom_paid →
  d = per_person - dorothy_paid →
  s = sammy_paid - per_person →
  t - d + s = 20 := by
  sorry

end vacation_cost_equality_l106_10646


namespace line_equation_l106_10640

/-- A line passing through a point with given intercepts -/
structure Line where
  point : ℝ × ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfies_equation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if an equation represents the given line -/
def is_equation_of_line (l : Line) (eq : LineEquation) : Prop :=
  satisfies_equation l.point eq ∧
  (eq.a ≠ 0 → satisfies_equation (l.x_intercept, 0) eq) ∧
  (eq.b ≠ 0 → satisfies_equation (0, l.y_intercept) eq)

/-- The main theorem -/
theorem line_equation (l : Line) 
    (h1 : l.point = (1, 2))
    (h2 : l.x_intercept = 2 * l.y_intercept) :
  (is_equation_of_line l ⟨2, -1, 0⟩) ∨ 
  (is_equation_of_line l ⟨1, 2, -5⟩) := by
  sorry

end line_equation_l106_10640


namespace new_person_weight_l106_10670

/-- The weight of the new person given the conditions of the problem -/
def weight_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating that the weight of the new person is 87.5 kg -/
theorem new_person_weight :
  weight_new_person 9 2.5 65 = 87.5 := by
  sorry

end new_person_weight_l106_10670


namespace f_has_at_most_two_zeros_l106_10696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + a

-- State the theorem
theorem f_has_at_most_two_zeros (a : ℝ) (h : a ≥ 16) :
  ∃ (z₁ z₂ : ℝ), ∀ x : ℝ, f a x = 0 → x = z₁ ∨ x = z₂ :=
sorry

end f_has_at_most_two_zeros_l106_10696


namespace initial_salt_concentration_l106_10644

/-- Given a salt solution that is diluted, proves the initial salt concentration --/
theorem initial_salt_concentration
  (initial_volume : ℝ)
  (water_added : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 90)
  (h2 : water_added = 30)
  (h3 : final_concentration = 0.15)
  : ∃ (initial_concentration : ℝ),
    initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) ∧
    initial_concentration = 0.2 := by
  sorry

end initial_salt_concentration_l106_10644


namespace expand_product_l106_10635

theorem expand_product (x : ℝ) : (x^2 - 3*x + 4) * (x^2 + 3*x + 1) = x^4 - 4*x^2 + 9*x + 4 := by
  sorry

end expand_product_l106_10635


namespace min_vertices_blue_triangle_or_red_K4_l106_10639

/-- A type representing a 2-coloring of edges in a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate for the existence of a blue triangle in a 2-coloring -/
def has_blue_triangle (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    c i j = false ∧ c j k = false ∧ c i k = false

/-- Predicate for the existence of a red K4 in a 2-coloring -/
def has_red_K4 (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    c i j = true ∧ c i k = true ∧ c i l = true ∧
    c j k = true ∧ c j l = true ∧ c k l = true

/-- The main theorem -/
theorem min_vertices_blue_triangle_or_red_K4 :
  (∀ n < 9, ∃ c : TwoColoring n, ¬has_blue_triangle n c ∧ ¬has_red_K4 n c) ∧
  (∀ c : TwoColoring 9, has_blue_triangle 9 c ∨ has_red_K4 9 c) :=
sorry

end min_vertices_blue_triangle_or_red_K4_l106_10639


namespace original_number_l106_10605

theorem original_number (x : ℝ) : ((x - 8 + 7) / 5) * 4 = 16 → x = 21 := by
  sorry

end original_number_l106_10605


namespace no_common_points_implies_b_range_l106_10679

theorem no_common_points_implies_b_range 
  (f g : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x = x^2 - a*x) 
  (k : ∀ x : ℝ, g x = b + a * Real.log (x - 1)) 
  (a_ge_one : a ≥ 1) 
  (no_common_points : ∀ x : ℝ, f x ≠ g x) : 
  b < 3/4 + Real.log 2 := by
sorry

end no_common_points_implies_b_range_l106_10679


namespace smallest_positive_a_quartic_polynomial_l106_10680

theorem smallest_positive_a_quartic_polynomial (a b : ℝ) : 
  (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - a*x + a = 0 → x > 0) →
  (∀ c : ℝ, c > 0 → (∃ d : ℝ, ∀ x : ℝ, x^4 - c*x^3 + d*x^2 - c*x + c = 0 → x > 0) → c ≥ a) →
  b = 6 * (4^(1/3))^2 :=
sorry

end smallest_positive_a_quartic_polynomial_l106_10680


namespace bobbit_worm_consumption_l106_10624

/-- Represents the number of fish eaten by the Bobbit worm each day -/
def fish_eaten_per_day : ℕ := 2

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish added after two weeks -/
def fish_added : ℕ := 8

/-- The number of fish remaining after three weeks -/
def remaining_fish : ℕ := 26

/-- The total number of days -/
def total_days : ℕ := 21

theorem bobbit_worm_consumption :
  initial_fish + fish_added - (fish_eaten_per_day * total_days) = remaining_fish := by
  sorry

end bobbit_worm_consumption_l106_10624


namespace absolute_value_of_x_minus_five_l106_10663

theorem absolute_value_of_x_minus_five (x : ℝ) (h : x = 4) : |x - 5| = 1 := by
  sorry

end absolute_value_of_x_minus_five_l106_10663


namespace smallest_number_with_same_prime_factors_l106_10626

def alice_number : ℕ := 60

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ (bob_number : ℕ), 
    has_all_prime_factors alice_number bob_number ∧
    ∀ (m : ℕ), has_all_prime_factors alice_number m → bob_number ≤ m ∧
    bob_number = 30 :=
sorry

end smallest_number_with_same_prime_factors_l106_10626


namespace percent_equality_l106_10632

theorem percent_equality (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end percent_equality_l106_10632


namespace lucille_weeding_ratio_l106_10673

def weed_value : ℕ := 6
def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def total_grass_weeds : ℕ := 32
def soda_cost : ℕ := 99
def remaining_money : ℕ := 147

theorem lucille_weeding_ratio :
  let total_earned := remaining_money + soda_cost
  let flower_veg_earnings := (flower_bed_weeds + vegetable_patch_weeds) * weed_value
  let grass_earnings := total_earned - flower_veg_earnings
  let grass_weeds_pulled := grass_earnings / weed_value
  (grass_weeds_pulled : ℚ) / total_grass_weeds = 1 / 2 :=
sorry

end lucille_weeding_ratio_l106_10673


namespace chocolate_eating_impossibility_l106_10607

/-- Proves that it's impossible to eat enough of the remaining chocolates to reach 3/2 of all chocolates eaten --/
theorem chocolate_eating_impossibility (total : ℕ) (initial_percent : ℚ) : 
  total = 10000 →
  initial_percent = 1/5 →
  ¬∃ (remaining_percent : ℚ), 
    0 ≤ remaining_percent ∧ 
    remaining_percent ≤ 1 ∧
    (initial_percent * total + remaining_percent * (total - initial_percent * total) : ℚ) = 3/2 * total := by
  sorry


end chocolate_eating_impossibility_l106_10607


namespace campaign_donation_proof_l106_10685

theorem campaign_donation_proof (max_donors : ℕ) (half_donors : ℕ) (total_raised : ℚ) 
  (h1 : max_donors = 500)
  (h2 : half_donors = 3 * max_donors)
  (h3 : total_raised = 3750000)
  (h4 : (max_donors * x + half_donors * (x / 2)) / total_raised = 2 / 5) :
  x = 1200 :=
by
  sorry

end campaign_donation_proof_l106_10685


namespace smallest_integer_fraction_sum_l106_10610

theorem smallest_integer_fraction_sum (A B C D : ℕ) : 
  A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (A + B) % (C + D) = 0 →
  (∀ E F G H : ℕ, E ≤ 9 ∧ F ≤ 9 ∧ G ≤ 9 ∧ H ≤ 9 →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    (E + F) % (G + H) = 0 →
    (A + B) / (C + D) ≤ (E + F) / (G + H)) →
  C + D = 17 :=
by sorry

end smallest_integer_fraction_sum_l106_10610


namespace unique_reverse_difference_l106_10633

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

/-- Checks if a number is a 4-digit number -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_reverse_difference :
  ∃! n : ℕ, isFourDigit n ∧ reverseDigits n = n + 8802 :=
by
  -- The proof goes here
  sorry

end unique_reverse_difference_l106_10633


namespace roots_properties_l106_10674

theorem roots_properties (x : ℝ) : 
  (x^2 - 7 * |x| + 6 = 0) → 
  (∃ (roots : Finset ℝ), 
    (∀ r ∈ roots, r^2 - 7 * |r| + 6 = 0) ∧ 
    (Finset.sum roots id = 0) ∧ 
    (Finset.prod roots id = 36)) :=
by sorry

end roots_properties_l106_10674


namespace functional_equation_solution_functional_equation_continuous_solution_l106_10623

def functional_equation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem functional_equation_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, functional_equation f a) ↔ (a = 0 ∨ a = 1/2 ∨ a = 1) :=
sorry

theorem functional_equation_continuous_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ functional_equation f a) ↔ a = 1/2 :=
sorry

end functional_equation_solution_functional_equation_continuous_solution_l106_10623


namespace angle_relations_l106_10601

theorem angle_relations (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2) 
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 4 / 3)
  (h_sin_diff : Real.sin (α - β) = -(Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ 
  Real.tan (α + β) = -41 / 38 := by
sorry

end angle_relations_l106_10601


namespace fred_total_games_l106_10668

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 36

/-- The number of basketball games Fred attended last year -/
def games_last_year : ℕ := 11

/-- The total number of basketball games Fred attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem fred_total_games : total_games = 47 := by
  sorry

end fred_total_games_l106_10668


namespace exists_empty_selection_l106_10630

/-- Represents a chessboard with pieces -/
structure Chessboard (n : ℕ) :=
  (board : Fin (2*n) → Fin (2*n) → Bool)
  (piece_count : Nat)
  (piece_count_eq : piece_count = 3*n)

/-- Represents a selection of rows and columns -/
structure Selection (n : ℕ) :=
  (rows : Fin n → Fin (2*n))
  (cols : Fin n → Fin (2*n))

/-- Checks if a selection results in an empty n × n chessboard -/
def is_empty_selection (cb : Chessboard n) (sel : Selection n) : Prop :=
  ∀ i j, ¬(cb.board (sel.rows i) (sel.cols j))

/-- Main theorem: There exists a selection that results in an empty n × n chessboard -/
theorem exists_empty_selection (n : ℕ) (cb : Chessboard n) :
  ∃ (sel : Selection n), is_empty_selection cb sel :=
sorry

end exists_empty_selection_l106_10630


namespace distribution_count_l106_10642

def number_of_women : ℕ := 2
def number_of_men : ℕ := 10
def number_of_magazines : ℕ := 8
def number_of_newspapers : ℕ := 4

theorem distribution_count :
  (Nat.choose number_of_men (number_of_newspapers - 1)) +
  (Nat.choose number_of_men number_of_newspapers) = 255 := by
  sorry

end distribution_count_l106_10642


namespace randys_initial_amount_l106_10648

/-- Proves that Randy's initial amount was $3000 given the described transactions --/
theorem randys_initial_amount (initial final smith_gave sally_received : ℕ) :
  final = initial + smith_gave - sally_received →
  smith_gave = 200 →
  sally_received = 1200 →
  final = 2000 →
  initial = 3000 := by
  sorry

end randys_initial_amount_l106_10648


namespace correct_equation_l106_10677

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * a^2 * b = -a^2 * b := by
  sorry

end correct_equation_l106_10677


namespace w_plus_reciprocal_w_traces_ellipse_l106_10687

theorem w_plus_reciprocal_w_traces_ellipse :
  ∀ (w : ℂ) (x y : ℝ),
  (Complex.abs w = 3) →
  (w + w⁻¹ = x + y * Complex.I) →
  ∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ a ≠ b ∧ a > 0 ∧ b > 0) := by
  sorry

end w_plus_reciprocal_w_traces_ellipse_l106_10687


namespace fraction_removal_sum_one_l106_10621

theorem fraction_removal_sum_one :
  let fractions : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let removed : List ℚ := [1/8, 1/10]
  let remaining : List ℚ := fractions.filter (fun x => x ∉ removed)
  remaining.sum = 1 := by
  sorry

end fraction_removal_sum_one_l106_10621


namespace square_sum_triples_l106_10645

theorem square_sum_triples :
  ∀ a b c : ℝ,
  (a = (b + c)^2 ∧ b = (a + c)^2 ∧ c = (a + b)^2) →
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/4 ∧ b = 1/4 ∧ c = 1/4)) :=
by sorry

end square_sum_triples_l106_10645


namespace wheat_profit_percentage_l106_10658

/-- Calculates the profit percentage for wheat mixture sales --/
theorem wheat_profit_percentage
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (selling_price : ℝ)
  (h1 : weight1 = 30)
  (h2 : price1 = 11.5)
  (h3 : weight2 = 20)
  (h4 : price2 = 14.25)
  (h5 : selling_price = 17.01) :
  let total_cost := weight1 * price1 + weight2 * price2
  let total_weight := weight1 + weight2
  let cost_per_kg := total_cost / total_weight
  let total_selling_price := selling_price * total_weight
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 35) < ε :=
by sorry

end wheat_profit_percentage_l106_10658


namespace unique_eventually_one_l106_10690

def f (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 1 else 3 * n

def eventually_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (f^[k] n) = 1

theorem unique_eventually_one :
  ∃! n : ℕ, n ∈ Finset.range 200 ∧ eventually_one n :=
sorry

end unique_eventually_one_l106_10690


namespace john_boxes_l106_10699

theorem john_boxes (stan_boxes : ℕ) (joseph_percent : ℚ) (jules_more : ℕ) (john_percent : ℚ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_percent = 80/100)
  (h3 : jules_more = 5)
  (h4 : john_percent = 20/100) :
  let joseph_boxes := stan_boxes * (1 - joseph_percent)
  let jules_boxes := joseph_boxes + jules_more
  let john_boxes := jules_boxes * (1 + john_percent)
  john_boxes = 30 := by sorry

end john_boxes_l106_10699


namespace sue_dog_walking_charge_l106_10664

/-- The amount Sue charged per dog for walking --/
def sue_charge_per_dog (perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed : ℚ) : ℚ :=
  let christian_total := christian_initial + christian_yards * christian_yard_price
  let initial_total := christian_total + sue_initial
  let needed := perfume_cost - initial_total
  let sue_earned := needed - additional_needed
  sue_earned / sue_dogs

theorem sue_dog_walking_charge 
  (perfume_cost : ℚ)
  (christian_initial : ℚ)
  (sue_initial : ℚ)
  (christian_yards : ℚ)
  (christian_yard_price : ℚ)
  (sue_dogs : ℚ)
  (additional_needed : ℚ)
  (h1 : perfume_cost = 50)
  (h2 : christian_initial = 5)
  (h3 : sue_initial = 7)
  (h4 : christian_yards = 4)
  (h5 : christian_yard_price = 5)
  (h6 : sue_dogs = 6)
  (h7 : additional_needed = 6) :
  sue_charge_per_dog perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed = 2 :=
by sorry

end sue_dog_walking_charge_l106_10664


namespace fred_basketball_games_l106_10602

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The difference in games between last year and this year -/
def game_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - game_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end fred_basketball_games_l106_10602


namespace sheets_per_box_l106_10643

theorem sheets_per_box (total_sheets : ℕ) (num_boxes : ℕ) (h1 : total_sheets = 700) (h2 : num_boxes = 7) :
  total_sheets / num_boxes = 100 := by
  sorry

end sheets_per_box_l106_10643


namespace distance_to_work_l106_10651

-- Define the problem parameters
def speed_to_work : ℝ := 45
def speed_from_work : ℝ := 30
def total_commute_time : ℝ := 1

-- Define the theorem
theorem distance_to_work :
  ∃ (d : ℝ), d / speed_to_work + d / speed_from_work = total_commute_time ∧ d = 18 :=
by
  sorry


end distance_to_work_l106_10651


namespace fifth_term_is_123_40_l106_10650

-- Define the arithmetic sequence
def arithmeticSequence (x y : ℚ) : ℕ → ℚ
  | 0 => x + y
  | 1 => x - y
  | 2 => x * y
  | 3 => x / y
  | n + 4 => arithmeticSequence x y 3 - (n + 1) * (2 * y)

-- Theorem statement
theorem fifth_term_is_123_40 (x y : ℚ) :
  x - y - (x + y) = -2 * y →
  x - 3 * y = x * y →
  x - 5 * y = x / y →
  y ≠ 0 →
  arithmeticSequence x y 4 = 123 / 40 :=
by sorry

end fifth_term_is_123_40_l106_10650


namespace correct_product_l106_10672

def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : Nat) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (0 < b) →             -- b is positive
  ((reverse_digits a) * b = 143) →  -- erroneous product
  (a * b = 341) :=
by sorry

end correct_product_l106_10672


namespace division_problem_l106_10689

theorem division_problem (n : ℚ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end division_problem_l106_10689


namespace last_released_theorem_l106_10667

/-- The position of the last released captive's servant -/
def last_released_position (N : ℕ) (total_purses : ℕ) : Set ℕ :=
  if total_purses = N + (N - 1) * N / 2
  then {N}
  else if total_purses = N + (N - 1) * N / 2 - 1
  then {N - 1, N}
  else ∅

/-- The main theorem about the position of the last released captive's servant -/
theorem last_released_theorem (N : ℕ) (total_purses : ℕ) 
  (h1 : N > 0) 
  (h2 : total_purses ≥ N) 
  (h3 : total_purses ≤ N + (N - 1) * N / 2) :
  (last_released_position N total_purses).Nonempty := by
  sorry

end last_released_theorem_l106_10667


namespace certain_point_on_circle_l106_10683

/-- A point on the parabola y^2 = 8x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- A circle with center on the parabola y^2 = 8x and tangent to x = -2 -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  is_tangent : center.x + radius = 2  -- Distance from center to x = -2 is equal to radius

theorem certain_point_on_circle (c : TangentCircle) : 
  (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

#check certain_point_on_circle

end certain_point_on_circle_l106_10683


namespace chennys_friends_l106_10691

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) : 
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end chennys_friends_l106_10691


namespace smallest_coin_set_l106_10636

/-- Represents the different types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A function that checks if a given set of coins can make all amounts from 1 to 99 cents --/
def canMakeAllAmounts (coins : List Coin) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount ≤ 99 →
    ∃ (subset : List Coin), subset ⊆ coins ∧ (subset.map coinValue).sum = amount

/-- The theorem stating that 6 is the smallest number of coins needed --/
theorem smallest_coin_set :
  ∃ (coins : List Coin),
    coins.length = 6 ∧
    canMakeAllAmounts coins ∧
    ∀ (other_coins : List Coin),
      canMakeAllAmounts other_coins →
      other_coins.length ≥ 6 :=
by sorry

#check smallest_coin_set

end smallest_coin_set_l106_10636


namespace curve_C_not_centrally_symmetric_l106_10634

-- Define the curve C
def C : ℝ → ℝ := fun x ↦ x^3 - x + 2

-- Theorem statement
theorem curve_C_not_centrally_symmetric :
  ∀ (a b : ℝ), ¬(∀ (x y : ℝ), C x = y → C (2*a - x) = 2*b - y) :=
by sorry

end curve_C_not_centrally_symmetric_l106_10634


namespace cubic_sum_over_product_l106_10660

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x * y + x * z + y * z ≠ 0) :
  (x^3 + y^3 + z^3) / (x * y * z * (x * y + x * z + y * z)) = -3 :=
sorry

end cubic_sum_over_product_l106_10660


namespace triangle_sine_inequality_l106_10619

theorem triangle_sine_inequality (A B C : Real) (h_triangle : A + B + C = π) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
  sorry

end triangle_sine_inequality_l106_10619


namespace eight_lines_divide_plane_into_37_regions_l106_10697

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- Theorem stating that 8 lines divide the plane into 37 regions -/
theorem eight_lines_divide_plane_into_37_regions :
  num_regions 8 = 37 := by sorry

end eight_lines_divide_plane_into_37_regions_l106_10697


namespace reading_time_calculation_l106_10686

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total reading time in hours -/
def total_reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_calculation :
  total_reading_time = 50 := by sorry

end reading_time_calculation_l106_10686


namespace parallel_sides_implies_parallelogram_l106_10641

/-- A quadrilateral is defined as a polygon with four sides -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- Parallel sides in a quadrilateral -/
def parallel_sides (q : Quadrilateral) (side1 side2 : Fin 4) : Prop :=
  -- Definition of parallel sides omitted for brevity
  sorry

/-- A parallelogram is a quadrilateral with both pairs of opposite sides parallel -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  parallel_sides q 0 2 ∧ parallel_sides q 1 3

/-- Theorem: If both pairs of opposite sides of a quadrilateral are parallel, then it is a parallelogram -/
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  (parallel_sides q 0 2 ∧ parallel_sides q 1 3) → is_parallelogram q :=
sorry

end parallel_sides_implies_parallelogram_l106_10641


namespace dividend_calculation_l106_10682

/-- Calculate the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.05) :
  let actual_share_price := share_value * (1 + premium_rate)
  let num_shares := investment / actual_share_price
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share = 600 := by
  sorry

end dividend_calculation_l106_10682


namespace product_scaled_down_l106_10678

theorem product_scaled_down (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 := by
  sorry

end product_scaled_down_l106_10678


namespace volunteers_needed_l106_10600

/-- Represents the number of volunteers needed for the school Christmas play --/
def total_volunteers_needed : ℕ := 100

/-- Represents the number of math classes --/
def math_classes : ℕ := 5

/-- Represents the number of students volunteering from each math class --/
def students_per_class : ℕ := 4

/-- Represents the total number of teachers volunteering --/
def teachers_volunteering : ℕ := 10

/-- Represents the number of teachers skilled in carpentry --/
def teachers_carpentry : ℕ := 3

/-- Represents the total number of parents volunteering --/
def parents_volunteering : ℕ := 15

/-- Represents the number of parents experienced with lighting and sound --/
def parents_lighting_sound : ℕ := 6

/-- Represents the additional number of volunteers needed with carpentry skills --/
def additional_carpentry_needed : ℕ := 8

/-- Represents the additional number of volunteers needed with lighting and sound experience --/
def additional_lighting_sound_needed : ℕ := 10

/-- Theorem stating that 9 more volunteers are needed to meet the requirements --/
theorem volunteers_needed : 
  (math_classes * students_per_class + teachers_volunteering + parents_volunteering) +
  (additional_carpentry_needed - teachers_carpentry) + 
  (additional_lighting_sound_needed - parents_lighting_sound) = 9 := by
  sorry

end volunteers_needed_l106_10600


namespace line_parametric_to_standard_l106_10638

/-- Given a line with parametric equations x = 1 + t and y = -1 + t,
    prove that its standard equation is x - y - 2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (x y t : ℝ), x = 1 + t ∧ y = -1 + t → x - y - 2 = 0 := by
sorry

end line_parametric_to_standard_l106_10638


namespace smallest_integer_m_l106_10617

theorem smallest_integer_m (x y m : ℝ) : 
  (3 * x + y = m + 8) → 
  (2 * x + 2 * y = 2 * m + 5) → 
  (x - y < 1) → 
  (∀ k : ℤ, k ≥ m → k ≥ 3) ∧ (3 : ℝ) ≥ m :=
by sorry

end smallest_integer_m_l106_10617


namespace line_passes_through_fixed_point_l106_10692

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + 2 * m + 1 = 1) := by
  sorry

end line_passes_through_fixed_point_l106_10692


namespace gcd_lcm_sum_l106_10606

theorem gcd_lcm_sum : Nat.gcd 28 63 + Nat.lcm 18 24 = 79 := by
  sorry

end gcd_lcm_sum_l106_10606


namespace range_of_s_l106_10681

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := sorry

-- State the theorem
theorem range_of_s :
  (∀ n ∈ CompositePositiveIntegers, s n > 7) ∧
  (∀ k > 7, ∃ n ∈ CompositePositiveIntegers, s n = k) :=
sorry

end range_of_s_l106_10681


namespace initial_rulers_count_l106_10604

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 34

/-- The number of rulers taken out of the drawer -/
def rulers_taken : ℕ := 11

/-- The number of rulers remaining in the drawer after removal -/
def rulers_remaining : ℕ := 3

theorem initial_rulers_count : initial_rulers = 14 := by sorry

end initial_rulers_count_l106_10604


namespace library_books_total_l106_10694

theorem library_books_total (initial_books additional_books : ℕ) 
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  initial_books + additional_books = 77 := by
sorry

end library_books_total_l106_10694


namespace variance_of_X_l106_10653

/-- A random variable X with two possible values -/
def X : Fin 2 → ℝ
  | 0 => 0
  | 1 => 1

/-- The probability mass function of X -/
def P : Fin 2 → ℝ
  | 0 => 0.4
  | 1 => 0.6

/-- The expected value of X -/
def E : ℝ := X 0 * P 0 + X 1 * P 1

/-- The variance of X -/
def D : ℝ := (X 0 - E)^2 * P 0 + (X 1 - E)^2 * P 1

/-- Theorem: The variance of X is 0.24 -/
theorem variance_of_X : D = 0.24 := by
  sorry

end variance_of_X_l106_10653


namespace real_roots_condition_l106_10675

theorem real_roots_condition (a b : ℝ) : 
  (∃ x : ℝ, (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1) ↔ 
  (1 / 2 < a / b ∧ a / b < 1) :=
sorry

end real_roots_condition_l106_10675


namespace smallest_prime_factors_sum_of_286_l106_10618

theorem smallest_prime_factors_sum_of_286 : 
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p < q ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 286 → r = p ∨ r = q ∨ r > q) ∧ 
    p + q = 13 := by
  sorry

end smallest_prime_factors_sum_of_286_l106_10618


namespace ellipse_regions_l106_10654

/-- 
Given n ellipses in a plane where:
- Any two ellipses intersect at exactly two points
- No three ellipses intersect at the same point

The number of regions these ellipses divide the plane into is n(n-1) + 2.
-/
theorem ellipse_regions (n : ℕ) : ℕ := by
  sorry

#check ellipse_regions

end ellipse_regions_l106_10654


namespace rain_thunder_prob_is_correct_l106_10698

/-- The probability of rain with thunder on both Monday and Tuesday -/
def rain_thunder_prob : ℝ :=
  let rain_monday_prob : ℝ := 0.40
  let rain_tuesday_prob : ℝ := 0.30
  let thunder_given_rain_prob : ℝ := 0.10
  let rain_both_days_prob : ℝ := rain_monday_prob * rain_tuesday_prob
  let thunder_both_days_given_rain_prob : ℝ := thunder_given_rain_prob * thunder_given_rain_prob
  rain_both_days_prob * thunder_both_days_given_rain_prob * 100

theorem rain_thunder_prob_is_correct : rain_thunder_prob = 0.12 := by
  sorry

end rain_thunder_prob_is_correct_l106_10698


namespace sum_of_common_ratios_l106_10665

/-- Given two nonconstant geometric sequences with terms k, a₁, a₂ and k, b₁, b₂ respectively,
    with different common ratios p and r, if a₂-b₂=5(a₁-b₁), then p + r = 5. -/
theorem sum_of_common_ratios (k p r : ℝ) (h_p_neq_r : p ≠ r) (h_p_neq_1 : p ≠ 1) (h_r_neq_1 : r ≠ 1)
    (h_eq : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end sum_of_common_ratios_l106_10665


namespace line_passes_through_fixed_point_l106_10659

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) :
  a * 1 + b * (-1) + c = 0 := by
sorry

end line_passes_through_fixed_point_l106_10659


namespace lost_money_proof_l106_10611

def money_lost (initial_amount spent_amount remaining_amount : ℕ) : ℕ :=
  (initial_amount - spent_amount) - remaining_amount

theorem lost_money_proof (initial_amount spent_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  money_lost initial_amount spent_amount remaining_amount = 6 := by
  sorry

end lost_money_proof_l106_10611


namespace midpoint_distance_to_y_axis_l106_10649

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h_A : point_on_parabola A) 
  (h_B : point_on_parabola B) 
  (h_dist : distance A focus + distance B focus = 12) :
  (A.1 + B.1) / 2 = 5 := by sorry

end

end midpoint_distance_to_y_axis_l106_10649
