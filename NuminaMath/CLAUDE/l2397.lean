import Mathlib

namespace rectangle_area_l2397_239785

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2025
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 18 * b :=
by sorry

end rectangle_area_l2397_239785


namespace range_of_fraction_l2397_239761

theorem range_of_fraction (x y : ℝ) (h : x^2 + y^2 + 2*x = 0) :
  ∃ (t : ℝ), y / (x - 1) = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
sorry

end range_of_fraction_l2397_239761


namespace polynomial_equation_solution_l2397_239710

theorem polynomial_equation_solution (x : ℝ) : 
  let q : ℝ → ℝ := λ t => 12 * t^3 - 4
  q (x^3) - q (x^3 - 4) = (q x)^2 + 20 := by
  sorry

end polynomial_equation_solution_l2397_239710


namespace cost_of_12_pencils_9_notebooks_l2397_239799

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 6 notebooks cost $3.21 -/
axiom condition1 : 9 * pencil_cost + 6 * notebook_cost = 3.21

/-- The second given condition: 8 pencils and 5 notebooks cost $2.84 -/
axiom condition2 : 8 * pencil_cost + 5 * notebook_cost = 2.84

/-- Theorem: The cost of 12 pencils and 9 notebooks is $4.32 -/
theorem cost_of_12_pencils_9_notebooks : 
  12 * pencil_cost + 9 * notebook_cost = 4.32 := by sorry

end cost_of_12_pencils_9_notebooks_l2397_239799


namespace gift_wrapping_combinations_l2397_239734

theorem gift_wrapping_combinations : 
  let wrapping_paper := 8
  let ribbon := 5
  let gift_card := 4
  let gift_sticker := 6
  wrapping_paper * ribbon * gift_card * gift_sticker = 960 := by
  sorry

end gift_wrapping_combinations_l2397_239734


namespace gcf_twenty_pair_l2397_239751

theorem gcf_twenty_pair : ∃! (a b : ℕ), 
  ((a = 200 ∧ b = 2000) ∨ 
   (a = 40 ∧ b = 50) ∨ 
   (a = 20 ∧ b = 40) ∨ 
   (a = 20 ∧ b = 25)) ∧ 
  Nat.gcd a b = 20 :=
by sorry

end gcf_twenty_pair_l2397_239751


namespace sachin_age_l2397_239714

theorem sachin_age : 
  ∀ (s r : ℕ), 
  r = s + 8 →  -- Sachin is younger than Rahul by 8 years
  s * 9 = r * 7 →  -- The ratio of their ages is 7 : 9
  s = 28 :=  -- Sachin's age is 28 years
by
  sorry

end sachin_age_l2397_239714


namespace wrap_vs_sleeve_difference_l2397_239700

def raw_squat : ℝ := 600
def sleeve_addition : ℝ := 30
def wrap_percentage : ℝ := 0.25

theorem wrap_vs_sleeve_difference :
  (raw_squat * wrap_percentage) - sleeve_addition = 120 := by
  sorry

end wrap_vs_sleeve_difference_l2397_239700


namespace cone_volume_l2397_239789

/-- Given a cone with slant height 3 and lateral area 3√5π, its volume is 10π/3 -/
theorem cone_volume (l : ℝ) (L : ℝ) (r : ℝ) (h : ℝ) (V : ℝ) : 
  l = 3 →
  L = 3 * Real.sqrt 5 * Real.pi →
  L = Real.pi * r * l →
  l^2 = r^2 + h^2 →
  V = (1/3) * Real.pi * r^2 * h →
  V = (10/3) * Real.pi := by
sorry


end cone_volume_l2397_239789


namespace coefficient_x4_in_expansion_l2397_239787

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 3)^5)
  ∃ (a b c d e f : ℤ), 
    (∀ x, expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) ∧
    b = 165 := by
  sorry

end coefficient_x4_in_expansion_l2397_239787


namespace max_value_fraction_l2397_239760

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 0.4 := by
sorry

end max_value_fraction_l2397_239760


namespace sum_of_specific_numbers_l2397_239736

theorem sum_of_specific_numbers : 1235 + 2351 + 3512 + 5123 = 12221 := by
  sorry

end sum_of_specific_numbers_l2397_239736


namespace paper_clip_collection_l2397_239731

theorem paper_clip_collection (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end paper_clip_collection_l2397_239731


namespace fraction_problem_l2397_239706

theorem fraction_problem (f : ℚ) : f * 16 + 5 = 13 → f = 1/2 := by
  sorry

end fraction_problem_l2397_239706


namespace percentage_equation_l2397_239756

theorem percentage_equation (x : ℝ) : (35 / 100 * 400 = 20 / 100 * x) → x = 700 := by
  sorry

end percentage_equation_l2397_239756


namespace product_of_negative_real_part_solutions_l2397_239772

theorem product_of_negative_real_part_solutions :
  let solutions : List (ℂ) := [2 * (Complex.exp (Complex.I * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 3 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 5 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 7 * Real.pi / 4))]
  let negative_real_part_solutions := solutions.filter (fun z => z.re < 0)
  ∀ z ∈ solutions, z^4 = -16 →
  negative_real_part_solutions.prod = 4 := by
sorry

end product_of_negative_real_part_solutions_l2397_239772


namespace complex_equation_solution_l2397_239716

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 5*I → z = 4/5 + 3/5*I := by
  sorry

end complex_equation_solution_l2397_239716


namespace polynomial_root_implies_h_value_l2397_239796

theorem polynomial_root_implies_h_value :
  ∀ h : ℝ, ((-2 : ℝ)^3 + h * (-2) - 12 = 0) → h = -10 := by
  sorry

end polynomial_root_implies_h_value_l2397_239796


namespace find_divisor_l2397_239737

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 25 := by
sorry

end find_divisor_l2397_239737


namespace square_perimeter_l2397_239769

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 500 / 3 → 
  area = side^2 → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 15 / 3 := by
  sorry

end square_perimeter_l2397_239769


namespace no_line_exists_l2397_239733

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line_through_focus m x y}

-- Define the distance from a point to the line x = -2
def distance_to_line (x y : ℝ) : ℝ := x + 2

-- Statement to prove
theorem no_line_exists :
  ¬ ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    distance_to_line A.1 A.2 + distance_to_line B.1 B.2 = 5 :=
sorry

end no_line_exists_l2397_239733


namespace panthers_score_l2397_239758

theorem panthers_score (total_points margin : ℕ) 
  (h1 : total_points = 34)
  (h2 : margin = 14) : 
  total_points - (total_points + margin) / 2 = 10 := by
  sorry

end panthers_score_l2397_239758


namespace work_completion_multiple_l2397_239705

/-- Given that some number of people can complete a work in 24 days,
    this theorem proves that 4 times that number of people
    can complete half the work in 6 days. -/
theorem work_completion_multiple :
  ∀ (P : ℕ) (W : ℝ),
  P > 0 →
  W > 0 →
  ∃ (m : ℕ),
    (P * 24 : ℝ) * W = (m * P * 6 : ℝ) * (W / 2) ∧
    m = 4 :=
by sorry

end work_completion_multiple_l2397_239705


namespace no_fibonacci_right_triangle_l2397_239740

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Theorem: No right-angled triangle has all sides as Fibonacci numbers -/
theorem no_fibonacci_right_triangle (n : ℕ) : 
  (fib n)^2 + (fib (n + 1))^2 ≠ (fib (n + 2))^2 := by
  sorry

end no_fibonacci_right_triangle_l2397_239740


namespace perfect_square_trinomial_m_values_l2397_239784

/-- A perfect square trinomial in the form ax^2 + bx + c -/
structure PerfectSquareTrinomial (a b c : ℝ) : Prop where
  is_perfect_square : ∃ (p q : ℝ), a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  PerfectSquareTrinomial 1 (m - 1) 9 → m = -5 ∨ m = 7 := by
  sorry

end perfect_square_trinomial_m_values_l2397_239784


namespace grape_juice_mixture_problem_l2397_239743

theorem grape_juice_mixture_problem (initial_volume : ℝ) (added_pure_juice : ℝ) (final_percentage : ℝ) :
  initial_volume = 30 →
  added_pure_juice = 10 →
  final_percentage = 0.325 →
  ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume + added_pure_juice = 
    (initial_volume + added_pure_juice) * final_percentage ∧
    initial_percentage = 0.1 := by
  sorry

end grape_juice_mixture_problem_l2397_239743


namespace function_value_at_two_l2397_239711

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end function_value_at_two_l2397_239711


namespace teacher_age_l2397_239776

/-- Given a class of students and their teacher, this theorem proves the teacher's age
    based on how the average age changes when including the teacher. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) 
    (h1 : num_students = 25)
    (h2 : student_avg_age = 26)
    (h3 : (num_students * student_avg_age + teacher_age) / (num_students + 1) = student_avg_age + 1) :
  teacher_age = 52 := by
  sorry

end teacher_age_l2397_239776


namespace dice_probability_l2397_239759

def standard_dice : ℕ := 5
def special_dice : ℕ := 5
def standard_sides : ℕ := 6
def special_sides : ℕ := 3  -- Only even numbers (2, 4, 6)

def probability_standard_one : ℚ := 1 / 6
def probability_standard_not_one : ℚ := 5 / 6
def probability_special_four : ℚ := 1 / 3
def probability_special_not_four : ℚ := 2 / 3

theorem dice_probability : 
  (Nat.choose standard_dice 1 : ℚ) * probability_standard_one * probability_standard_not_one ^ 4 *
  (Nat.choose special_dice 1 : ℚ) * probability_special_four * probability_special_not_four ^ 4 =
  250000 / 1889568 := by sorry

end dice_probability_l2397_239759


namespace gwen_science_problems_l2397_239703

/-- Given information about Gwen's homework problems, prove that she had 11 science problems. -/
theorem gwen_science_problems
  (math_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_problems : ℕ)
  (h1 : math_problems = 18)
  (h2 : finished_problems = 24)
  (h3 : remaining_problems = 5) :
  finished_problems + remaining_problems - math_problems = 11 :=
by sorry

end gwen_science_problems_l2397_239703


namespace jacob_dimes_l2397_239741

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coin_value "penny" + nickels * coin_value "nickel" + dimes * coin_value "dime"

theorem jacob_dimes (mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes : ℕ)
                    (jacob_pennies jacob_nickels : ℕ)
                    (difference : ℕ) :
  mrs_hilt_pennies = 2 →
  mrs_hilt_nickels = 2 →
  mrs_hilt_dimes = 2 →
  jacob_pennies = 4 →
  jacob_nickels = 1 →
  difference = 13 →
  ∃ jacob_dimes : ℕ,
    total_value mrs_hilt_pennies mrs_hilt_nickels mrs_hilt_dimes -
    total_value jacob_pennies jacob_nickels jacob_dimes = difference ∧
    jacob_dimes = 1 :=
by sorry

end jacob_dimes_l2397_239741


namespace wall_height_is_600_l2397_239742

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of a single brick -/
def brick_dim : Dimensions := ⟨80, 11.25, 6⟩

/-- The known dimensions of the wall (length and width) -/
def wall_dim (h : ℝ) : Dimensions := ⟨800, 22.5, h⟩

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that if 2000 bricks of given dimensions are required to build a wall
    with known length and width, then the height of the wall is 600 cm -/
theorem wall_height_is_600 :
  volume (wall_dim 600) = (volume brick_dim) * num_bricks := by sorry

end wall_height_is_600_l2397_239742


namespace shifted_roots_polynomial_l2397_239767

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) := by
sorry

end shifted_roots_polynomial_l2397_239767


namespace a_equals_one_necessary_not_sufficient_l2397_239721

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Statement of the theorem
theorem a_equals_one_necessary_not_sufficient :
  (∃ a : ℝ, a ≠ 1 ∧ A ∪ B a = Set.univ) ∧
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) :=
by sorry

end a_equals_one_necessary_not_sufficient_l2397_239721


namespace set_operations_and_intersection_l2397_239781

open Set

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  (Bᶜ = {x | x ≤ 2 ∨ x ≥ 10}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty → a > 1) := by
  sorry

end set_operations_and_intersection_l2397_239781


namespace deck_problem_l2397_239774

theorem deck_problem (r b : ℕ) : 
  r / (r + b) = 1 / 5 →
  r / (r + (b + 6)) = 1 / 7 →
  r = 3 :=
by sorry

end deck_problem_l2397_239774


namespace line_through_ellipse_midpoint_l2397_239727

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on the given ellipse -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 / 25 + p.y^2 / 16 = 1

/-- Checks if a point lies on the given line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

theorem line_through_ellipse_midpoint (M A B : Point) (l : Line) :
  isOnLine M l →
  isOnEllipse A →
  isOnEllipse B →
  isOnLine A l →
  isOnLine B l →
  isMidpoint M A B →
  M.x = 1 →
  M.y = 2 →
  l.a = 8 ∧ l.b = 25 ∧ l.c = -58 := by
  sorry


end line_through_ellipse_midpoint_l2397_239727


namespace seating_theorem_l2397_239725

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat athletes from three teams in a row, with teammates seated together -/
def seating_arrangements (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  factorial 3 * factorial team_a * factorial team_b * factorial team_c

theorem seating_theorem :
  seating_arrangements 4 3 3 = 5184 := by
  sorry

end seating_theorem_l2397_239725


namespace number_representation_proof_l2397_239792

theorem number_representation_proof (n a b c : ℕ) : 
  (n = 14^2 * a + 14 * b + c) →
  (n = 15^2 * a + 15 * c + b) →
  (n = 6^3 * a + 6^2 * c + 6 * a + c) →
  (a > 0) →
  (a < 6 ∧ b < 14 ∧ c < 6) →
  (n = 925) := by
sorry

end number_representation_proof_l2397_239792


namespace correct_operation_l2397_239770

theorem correct_operation (a b : ℝ) : 2 * a^2 * b * (4 * a * b^3) = 8 * a^3 * b^4 := by
  sorry

end correct_operation_l2397_239770


namespace light_flash_interval_l2397_239794

/-- Given a light that flashes 180 times in ¾ of an hour, 
    prove that the time between flashes is 15 seconds. -/
theorem light_flash_interval (flashes : ℕ) (time : ℚ) 
  (h1 : flashes = 180) 
  (h2 : time = 3/4) : 
  (time * 3600) / flashes = 15 := by
  sorry

end light_flash_interval_l2397_239794


namespace f_increasing_iff_a_in_interval_l2397_239779

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

/-- The theorem stating that f(x) is increasing on ℝ if and only if a ∈ [-1, 1] -/
theorem f_increasing_iff_a_in_interval (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end f_increasing_iff_a_in_interval_l2397_239779


namespace percentage_with_both_pets_l2397_239797

def total_students : ℕ := 40
def puppy_percentage : ℚ := 80 / 100
def both_pets : ℕ := 8

theorem percentage_with_both_pets : 
  (both_pets : ℚ) / (puppy_percentage * total_students) * 100 = 25 := by
  sorry

end percentage_with_both_pets_l2397_239797


namespace seonyeong_class_size_l2397_239755

/-- The number of rows of students -/
def num_rows : ℕ := 12

/-- The number of students in each row -/
def students_per_row : ℕ := 4

/-- The number of additional students -/
def additional_students : ℕ := 3

/-- The number of students in Jieun's class -/
def jieun_class_size : ℕ := 12

/-- The total number of students -/
def total_students : ℕ := num_rows * students_per_row + additional_students

/-- Theorem: The number of students in Seonyeong's class is 39 -/
theorem seonyeong_class_size : total_students - jieun_class_size = 39 := by
  sorry

end seonyeong_class_size_l2397_239755


namespace first_donor_amount_l2397_239722

theorem first_donor_amount (d1 d2 d3 d4 : ℝ) 
  (h1 : d2 = 2 * d1)
  (h2 : d3 = 3 * d2)
  (h3 : d4 = 4 * d3)
  (h4 : d1 + d2 + d3 + d4 = 132) :
  d1 = 4 := by
  sorry

end first_donor_amount_l2397_239722


namespace power_of_power_l2397_239754

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end power_of_power_l2397_239754


namespace quadratic_equation_properties_l2397_239739

/-- Given a quadratic equation kx^2 - 2(k+1)x + k-1 = 0 with two distinct real roots, 
    this theorem proves properties about the range of k and the sum of reciprocals of roots. -/
theorem quadratic_equation_properties (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 ∧ k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0) →
  (k > -1/3 ∧ k ≠ 0) ∧
  ¬(∃ k : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → k*x₁^2 - 2*(k+1)*x₁ + (k-1) = 0 → k*x₂^2 - 2*(k+1)*x₂ + (k-1) = 0 → 
    1/x₁ + 1/x₂ = 0) :=
by sorry

end quadratic_equation_properties_l2397_239739


namespace binomial_product_minus_240_l2397_239793

theorem binomial_product_minus_240 : 
  (Nat.choose 10 3) * (Nat.choose 8 3) - 240 = 6480 := by
  sorry

end binomial_product_minus_240_l2397_239793


namespace range_of_m_min_value_sum_squares_equality_condition_l2397_239748

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x ≤ -m^2 + 6*m) → 1 ≤ m ∧ m ≤ 5 :=
sorry

-- Theorem 2: Minimum value of a^2 + b^2 + c^2
theorem min_value_sum_squares (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 ≥ 1/2 :=
sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 5 →
  a^2 + b^2 + c^2 = 1/2 ↔ a = 3/10 ∧ b = 4/10 ∧ c = 5/10 :=
sorry

end range_of_m_min_value_sum_squares_equality_condition_l2397_239748


namespace line_parameter_range_l2397_239765

/-- Given two points on opposite sides of a line, prove the range of the line's parameter. -/
theorem line_parameter_range (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y + m = 0 → 
    ((x = 1 ∧ y = 3) ∨ (x = -4 ∧ y = -2)) →
    (2*1 + 3 + m) * (2*(-4) + (-2) + m) < 0) →
  -5 < m ∧ m < 10 :=
sorry

end line_parameter_range_l2397_239765


namespace abc_sum_difference_l2397_239753

theorem abc_sum_difference (a b c : ℝ) 
  (hab : |a - b| = 1)
  (hbc : |b - c| = 1)
  (hca : |c - a| = 2)
  (habc : a * b * c = 60) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 10 := by
  sorry

end abc_sum_difference_l2397_239753


namespace recurring_decimal_division_l2397_239763

theorem recurring_decimal_division :
  let a : ℚ := 36 / 99
  let b : ℚ := 12 / 99
  a / b = 3 := by sorry

end recurring_decimal_division_l2397_239763


namespace smallest_multiple_with_100_divisors_l2397_239783

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ m : ℕ,
    m > 0 ∧
    is_multiple m 100 ∧
    num_divisors m = 100 ∧
    (∀ k : ℕ, k > 0 → is_multiple k 100 → num_divisors k = 100 → m ≤ k) ∧
    m / 100 = 324 :=
sorry

end smallest_multiple_with_100_divisors_l2397_239783


namespace triangle_third_side_length_l2397_239720

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (γ : ℝ) 
  (ha : a = 7) 
  (hb : b = 8) 
  (hγ : γ = 2 * π / 3) -- 120° in radians
  (hc : c^2 = a^2 + b^2 - 2*a*b*Real.cos γ) : -- Law of Cosines
  c = 13 := by
sorry

end triangle_third_side_length_l2397_239720


namespace optimal_truck_loading_l2397_239791

theorem optimal_truck_loading (total_load : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (h_total : total_load = 134)
  (h_large : large_capacity = 15)
  (h_small : small_capacity = 7) :
  ∃ (large_count small_count : ℕ),
    large_count * large_capacity + small_count * small_capacity = total_load ∧
    large_count = 8 ∧
    small_count = 2 ∧
    ∀ (l s : ℕ), l * large_capacity + s * small_capacity = total_load →
      l + s ≥ large_count + small_count :=
by sorry

end optimal_truck_loading_l2397_239791


namespace percentage_equation_l2397_239744

theorem percentage_equation (x : ℝ) : 0.65 * x = 0.20 * 552.50 → x = 170 := by
  sorry

end percentage_equation_l2397_239744


namespace geometric_sequence_sum_l2397_239775

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →  -- First term
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- Arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l2397_239775


namespace square_formation_possible_l2397_239715

theorem square_formation_possible (figure_area : ℕ) (h : figure_area = 4) :
  ∃ (n : ℕ), n > 0 ∧ (n * n) % figure_area = 0 :=
sorry

end square_formation_possible_l2397_239715


namespace red_light_probability_l2397_239771

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle time
def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Define the probability of seeing a red light
def probability_red_light : ℚ := red_duration / total_cycle_time

-- Theorem statement
theorem red_light_probability :
  probability_red_light = 30 / 75 :=
by sorry

end red_light_probability_l2397_239771


namespace final_pen_count_l2397_239732

theorem final_pen_count (x : ℝ) (x_pos : x > 0) : 
  let after_mike := x + 0.5 * x
  let after_cindy := 2 * after_mike
  let given_to_sharon := 0.25 * after_cindy
  after_cindy - given_to_sharon = 2.25 * x :=
by sorry

end final_pen_count_l2397_239732


namespace minimize_S_l2397_239773

/-- The sum of squared differences function -/
def S (x y z : ℝ) : ℝ :=
  (x + y + z - 10)^2 + (x + y - z - 7)^2 + (x - y + z - 6)^2 + (-x + y + z - 5)^2

/-- Theorem stating that (4.5, 4, 3.5) minimizes S -/
theorem minimize_S :
  ∀ x y z : ℝ, S x y z ≥ S 4.5 4 3.5 := by sorry

end minimize_S_l2397_239773


namespace set_equality_l2397_239712

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem set_equality : (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end set_equality_l2397_239712


namespace cupboard_cost_price_l2397_239747

/-- The cost price of the cupboard -/
def cost_price : ℝ := sorry

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price -/
def increased_selling_price : ℝ := 1.16 * cost_price

theorem cupboard_cost_price : cost_price = 3750 := by
  have h1 : selling_price = 0.84 * cost_price := rfl
  have h2 : increased_selling_price = 1.16 * cost_price := rfl
  have h3 : increased_selling_price - selling_price = 1200 := sorry
  sorry

end cupboard_cost_price_l2397_239747


namespace reciprocal_of_negative_2023_l2397_239798

theorem reciprocal_of_negative_2023 :
  ∃ (x : ℚ), x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end reciprocal_of_negative_2023_l2397_239798


namespace degree_of_h_l2397_239746

/-- Given a polynomial f(x) = -5x^5 + 2x^4 + 7x - 8 and a polynomial h(x) such that
    the degree of f(x) - h(x) is 3, prove that the degree of h(x) is 5. -/
theorem degree_of_h (f h : Polynomial ℝ) : 
  f = -5 * X^5 + 2 * X^4 + 7 * X - 8 →
  Polynomial.degree (f - h) = 3 →
  Polynomial.degree h = 5 :=
by sorry

end degree_of_h_l2397_239746


namespace range_of_m_for_equation_l2397_239724

theorem range_of_m_for_equation (P : Prop) 
  (h : P ↔ ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) : 
  P → ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 := by
  sorry

end range_of_m_for_equation_l2397_239724


namespace right_triangle_exterior_angles_sum_l2397_239757

theorem right_triangle_exterior_angles_sum (α β γ δ ε : Real) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle
  γ = 90 →           -- Right angle in the triangle
  α + δ = 180 →      -- Linear pair for first non-right angle
  β + ε = 180 →      -- Linear pair for second non-right angle
  δ + ε = 270 :=     -- Sum of exterior angles
by sorry

end right_triangle_exterior_angles_sum_l2397_239757


namespace simplify_fraction_l2397_239795

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end simplify_fraction_l2397_239795


namespace cubic_root_sum_cubes_l2397_239728

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
sorry

end cubic_root_sum_cubes_l2397_239728


namespace parabola_intersection_l2397_239708

theorem parabola_intersection (m : ℝ) (h : m > 0) :
  let f (x : ℝ) := x^2 + 2*m*x - (5/4)*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 6 → m = 2) :=
by sorry

end parabola_intersection_l2397_239708


namespace cherry_soda_count_l2397_239768

theorem cherry_soda_count (total_cans : ℕ) (orange_ratio : ℕ) (cherry_count : ℕ) : 
  total_cans = 24 →
  orange_ratio = 2 →
  total_cans = cherry_count + orange_ratio * cherry_count →
  cherry_count = 8 := by
  sorry

end cherry_soda_count_l2397_239768


namespace cost_price_of_toy_cost_price_is_800_l2397_239788

/-- The cost price of a toy given the selling price and gain conditions -/
theorem cost_price_of_toy (total_sale : ℕ) (num_toys : ℕ) (gain_in_toys : ℕ) : ℕ :=
  let selling_price := total_sale / num_toys
  let cost_price := selling_price / (1 + gain_in_toys / num_toys)
  cost_price
  
/-- Proof that the cost price of a toy is 800 given the conditions -/
theorem cost_price_is_800 : cost_price_of_toy 16800 18 3 = 800 := by
  sorry

end cost_price_of_toy_cost_price_is_800_l2397_239788


namespace truncated_pyramid_surface_area_l2397_239780

/-- The total surface area of a truncated right pyramid with given dimensions --/
theorem truncated_pyramid_surface_area
  (base_side : ℝ)
  (upper_side : ℝ)
  (height : ℝ)
  (h_base : base_side = 15)
  (h_upper : upper_side = 10)
  (h_height : height = 20) :
  let slant_height := Real.sqrt (height^2 + ((base_side - upper_side) / 2)^2)
  let lateral_area := 2 * (base_side + upper_side) * slant_height
  let base_area := base_side^2 + upper_side^2
  lateral_area + base_area = 1332.8 :=
by sorry

end truncated_pyramid_surface_area_l2397_239780


namespace quadratic_minimum_less_than_neg_six_l2397_239702

/-- A quadratic function satisfying specific point conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-2) = 6 ∧ f 0 = -4 ∧ f 1 = -6 ∧ f 3 = -4

/-- The theorem stating that the minimum value of the quadratic function is less than -6 -/
theorem quadratic_minimum_less_than_neg_six (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x < -6 :=
sorry

end quadratic_minimum_less_than_neg_six_l2397_239702


namespace angle_Y_value_l2397_239735

-- Define the angles as real numbers
def A : ℝ := 50
def Z : ℝ := 50

-- Define the theorem
theorem angle_Y_value :
  ∀ B X Y : ℝ,
  A + B = 180 →
  X = Y →
  B + Z = 180 →
  B + X + Y = 180 →
  Y = 25 :=
by
  sorry

end angle_Y_value_l2397_239735


namespace trig_identity_l2397_239701

theorem trig_identity : Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end trig_identity_l2397_239701


namespace victor_score_l2397_239766

/-- 
Given a maximum mark and a percentage score, calculate the actual score.
-/
def calculateScore (maxMark : ℕ) (percentage : ℚ) : ℚ :=
  percentage * maxMark

theorem victor_score :
  let maxMark : ℕ := 300
  let percentage : ℚ := 80 / 100
  calculateScore maxMark percentage = 240 := by
  sorry

end victor_score_l2397_239766


namespace product_of_n_values_product_of_possible_n_values_l2397_239726

-- Define the temperatures at noon
def temp_minneapolis (n : ℝ) (l : ℝ) : ℝ := l + n
def temp_stlouis (l : ℝ) : ℝ := l

-- Define the temperatures at 4:00 PM
def temp_minneapolis_4pm (n : ℝ) (l : ℝ) : ℝ := temp_minneapolis n l - 7
def temp_stlouis_4pm (l : ℝ) : ℝ := temp_stlouis l + 5

-- Define the temperature difference at 4:00 PM
def temp_diff_4pm (n : ℝ) (l : ℝ) : ℝ := |temp_minneapolis_4pm n l - temp_stlouis_4pm l|

-- Theorem statement
theorem product_of_n_values (n : ℝ) (l : ℝ) :
  (temp_diff_4pm n l = 4) → (n = 16 ∨ n = 8) ∧ (16 * 8 = 128) := by
  sorry

-- Main theorem
theorem product_of_possible_n_values : 
  ∃ (n₁ n₂ : ℝ), (n₁ ≠ n₂) ∧ (∀ l : ℝ, temp_diff_4pm n₁ l = 4 ∧ temp_diff_4pm n₂ l = 4) ∧ (n₁ * n₂ = 128) := by
  sorry

end product_of_n_values_product_of_possible_n_values_l2397_239726


namespace wallet_theorem_l2397_239762

def wallet_problem (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) (twenty_dollar_bills : ℕ) : Prop :=
  let total_amount : ℕ := 150
  let ten_dollar_amount : ℕ := 50
  let twenty_dollar_count : ℕ := 4
  (5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills = total_amount) ∧
  (10 * ten_dollar_bills = ten_dollar_amount) ∧
  (twenty_dollar_bills = twenty_dollar_count) ∧
  (five_dollar_bills + ten_dollar_bills + twenty_dollar_bills = 13)

theorem wallet_theorem :
  ∃ (five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ),
    wallet_problem five_dollar_bills ten_dollar_bills twenty_dollar_bills :=
by
  sorry

end wallet_theorem_l2397_239762


namespace presents_difference_l2397_239730

def ethan_presents : ℝ := 31.0
def alissa_presents : ℕ := 9

theorem presents_difference : ethan_presents - alissa_presents = 22 := by
  sorry

end presents_difference_l2397_239730


namespace fourth_root_of_12960000_l2397_239704

theorem fourth_root_of_12960000 : Real.sqrt (Real.sqrt 12960000) = 60 := by
  sorry

end fourth_root_of_12960000_l2397_239704


namespace intersection_irrationality_l2397_239782

theorem intersection_irrationality (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ∀ x : ℚ, x^2 - 2*p*x + 2*q ≠ 0 :=
sorry

end intersection_irrationality_l2397_239782


namespace volunteer_selection_probability_l2397_239707

theorem volunteer_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2018) 
  (h2 : eliminated = 18) 
  (h3 : selected = 50) :
  (selected : ℚ) / total_students = 25 / 1009 := by
sorry

end volunteer_selection_probability_l2397_239707


namespace geometric_progression_common_ratio_l2397_239729

theorem geometric_progression_common_ratio 
  (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) 
  (h_geom_prog : ∃ (a r : ℝ), r ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2 ∧ 
    w * (x - y) = a * r^3) : 
  ∃ r : ℝ, r^3 + r^2 + r + 1 = 0 := by
sorry

end geometric_progression_common_ratio_l2397_239729


namespace polynomial_divisibility_l2397_239749

/-- If x^3 - 2x^2 + px + q is divisible by x + 2, then q = 16 + 2p -/
theorem polynomial_divisibility (p q : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^3 - 2*x^2 + p*x + q = (x + 2) * k) → 
  q = 16 + 2*p := by
sorry

end polynomial_divisibility_l2397_239749


namespace concert_ticket_cost_l2397_239764

theorem concert_ticket_cost (current_amount : ℕ) (amount_needed : ℕ) (num_tickets : ℕ) : 
  current_amount = 189 →
  amount_needed = 171 →
  num_tickets = 4 →
  (current_amount + amount_needed) / num_tickets = 90 := by
sorry

end concert_ticket_cost_l2397_239764


namespace equal_share_theorem_l2397_239752

/-- Represents the number of stickers each person has -/
structure Stickers where
  kate : ℝ
  jenna : ℝ
  ava : ℝ

/-- The ratio of stickers between Kate, Jenna, and Ava -/
def sticker_ratio : Stickers := { kate := 7.5, jenna := 4.25, ava := 5.75 }

/-- Kate's actual number of stickers -/
def kate_stickers : ℝ := 45

/-- Calculates the total number of stickers -/
def total_stickers (s : Stickers) : ℝ := s.kate + s.jenna + s.ava

/-- Theorem stating that when the stickers are equally shared, each person gets 35 stickers -/
theorem equal_share_theorem (s : Stickers) :
  s.kate / sticker_ratio.kate = s.jenna / sticker_ratio.jenna ∧
  s.kate / sticker_ratio.kate = s.ava / sticker_ratio.ava ∧
  s.kate = kate_stickers →
  (total_stickers s) / 3 = 35 := by sorry

end equal_share_theorem_l2397_239752


namespace G_n_planarity_l2397_239709

/-- A graph G_n where vertices are integers from 1 to n -/
def G_n (n : ℕ) := {v : ℕ // v ≤ n}

/-- Two vertices are connected if and only if their sum is prime -/
def connected (n : ℕ) (a b : G_n n) : Prop :=
  Nat.Prime (a.val + b.val)

/-- The graph G_n is planar -/
def is_planar (n : ℕ) : Prop :=
  ∃ (f : G_n n → ℝ × ℝ), ∀ (a b c d : G_n n),
    a ≠ b ∧ c ≠ d ∧ connected n a b ∧ connected n c d →
    (f a ≠ f c ∨ f b ≠ f d) ∧ (f a ≠ f d ∨ f b ≠ f c)

/-- The main theorem: G_n is planar if and only if n ≤ 8 -/
theorem G_n_planarity (n : ℕ) : is_planar n ↔ n ≤ 8 := by
  sorry

end G_n_planarity_l2397_239709


namespace coin_combinations_eq_20_l2397_239750

/-- The number of combinations of pennies, nickels, and quarters that sum to 50 cents -/
def coin_combinations : Nat :=
  (Finset.filter (fun (p, n, q) => p + 5 * n + 25 * q = 50)
    (Finset.product (Finset.range 51)
      (Finset.product (Finset.range 11) (Finset.range 3)))).card

/-- Theorem stating that the number of coin combinations is 20 -/
theorem coin_combinations_eq_20 : coin_combinations = 20 := by
  sorry

end coin_combinations_eq_20_l2397_239750


namespace pi_digits_ratio_l2397_239790

/-- The number of digits of pi memorized by Carlos -/
def carlos_digits : ℕ := sorry

/-- The number of digits of pi memorized by Sam -/
def sam_digits : ℕ := sorry

/-- The number of digits of pi memorized by Mina -/
def mina_digits : ℕ := sorry

/-- The ratio of digits memorized by Mina to Carlos -/
def mina_carlos_ratio : ℚ := sorry

theorem pi_digits_ratio :
  sam_digits = carlos_digits + 6 ∧
  mina_digits = 24 ∧
  sam_digits = 10 ∧
  ∃ k : ℕ, mina_digits = k * carlos_digits →
  mina_carlos_ratio = 6 := by sorry

end pi_digits_ratio_l2397_239790


namespace smallest_number_l2397_239717

/-- Converts a number from base b to decimal -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The given numbers in their respective bases -/
def number_a : List Nat := [3, 3]
def number_b : List Nat := [0, 1, 1, 1]
def number_c : List Nat := [2, 2, 1]
def number_d : List Nat := [1, 2]

theorem smallest_number :
  to_decimal number_d 5 < to_decimal number_a 4 ∧
  to_decimal number_d 5 < to_decimal number_b 2 ∧
  to_decimal number_d 5 < to_decimal number_c 3 :=
by sorry

end smallest_number_l2397_239717


namespace repeat2016_product_of_palindromes_l2397_239777

/-- A natural number is a palindrome if it reads the same forwards and backwards. -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- Repeats the digits 2016 n times to form a natural number. -/
def repeat2016 (n : ℕ) : ℕ := sorry

/-- Theorem: Any number formed by repeating 2016 n times is the product of two palindromes. -/
theorem repeat2016_product_of_palindromes (n : ℕ) (h : n ≥ 1) :
  ∃ (a b : ℕ), isPalindrome a ∧ isPalindrome b ∧ repeat2016 n = a * b := by sorry

end repeat2016_product_of_palindromes_l2397_239777


namespace leg_lengths_are_3_or_4_l2397_239723

/-- Represents an isosceles triangle with integer side lengths --/
structure IsoscelesTriangle where
  base : ℕ
  leg : ℕ
  sum_eq_10 : base + 2 * leg = 10

/-- The set of possible leg lengths for an isosceles triangle formed from a 10cm wire --/
def possible_leg_lengths : Set ℕ :=
  {l | ∃ t : IsoscelesTriangle, t.leg = l}

/-- Theorem stating that the only possible leg lengths are 3 and 4 --/
theorem leg_lengths_are_3_or_4 : possible_leg_lengths = {3, 4} := by
  sorry

#check leg_lengths_are_3_or_4

end leg_lengths_are_3_or_4_l2397_239723


namespace complex_norm_squared_l2397_239745

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 5 + 4*Complex.I) : 
  Complex.abs z^2 = 41/10 := by
sorry

end complex_norm_squared_l2397_239745


namespace farmer_feed_cost_l2397_239718

theorem farmer_feed_cost (total_spent : ℝ) (chicken_feed_percent : ℝ) (chicken_discount : ℝ) : 
  total_spent = 35 →
  chicken_feed_percent = 0.4 →
  chicken_discount = 0.5 →
  let chicken_feed_cost := total_spent * chicken_feed_percent
  let goat_feed_cost := total_spent * (1 - chicken_feed_percent)
  let full_price_chicken_feed := chicken_feed_cost / (1 - chicken_discount)
  let full_price_total := full_price_chicken_feed + goat_feed_cost
  full_price_total = 49 := by sorry

end farmer_feed_cost_l2397_239718


namespace water_consumption_rate_l2397_239738

/-- 
Given a person drinks water at a rate of 1 cup every 20 minutes,
prove that they will drink 11.25 cups in 225 minutes.
-/
theorem water_consumption_rate (drinking_rate : ℚ) (time : ℚ) (cups : ℚ) : 
  drinking_rate = 1 / 20 → time = 225 → cups = time * drinking_rate → cups = 11.25 := by
  sorry

end water_consumption_rate_l2397_239738


namespace cube_sum_is_18_l2397_239719

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 9

/-- The sum of numbers on a face of the cube -/
def face_sum (arrangement : CubeArrangement) (face : Finset (Fin 8)) : ℕ :=
  (face.sum fun v => arrangement v).val + 1

/-- Predicate for a valid cube arrangement -/
def is_valid_arrangement (arrangement : CubeArrangement) : Prop :=
  ∀ (face1 face2 : Finset (Fin 8)), face1.card = 4 → face2.card = 4 → 
    face_sum arrangement face1 = face_sum arrangement face2

theorem cube_sum_is_18 :
  ∀ (arrangement : CubeArrangement), is_valid_arrangement arrangement →
    ∃ (face : Finset (Fin 8)), face.card = 4 ∧ face_sum arrangement face = 18 :=
sorry

end cube_sum_is_18_l2397_239719


namespace ratio_to_percentage_increase_l2397_239713

theorem ratio_to_percentage_increase (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end ratio_to_percentage_increase_l2397_239713


namespace quadratic_roots_relation_l2397_239786

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) ∧
               (3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = p)) →
  p / q = 27 := by
sorry

end quadratic_roots_relation_l2397_239786


namespace scientific_notation_equals_original_number_l2397_239778

def scientific_notation_value : ℝ := 6.7 * (10 ^ 6)

theorem scientific_notation_equals_original_number : scientific_notation_value = 6700000 := by
  sorry

end scientific_notation_equals_original_number_l2397_239778
