import Mathlib

namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l1846_184660

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l1846_184660


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_result_l1846_184619

theorem complex_arithmetic_expression_result : 
  let expr := 3034 - ((1002 / 20.04) * (43.8 - 9.2^2) + Real.sqrt 144) / (3.58 * (76 - 8.23^3))
  ∃ ε > 0, abs (expr - 1.17857142857) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_result_l1846_184619


namespace NUMINAMATH_CALUDE_max_thursday_money_l1846_184689

def tuesday_amount : ℕ := 8

def wednesday_amount : ℕ := 5 * tuesday_amount

def thursday_amount : ℕ := tuesday_amount + 41

theorem max_thursday_money : thursday_amount = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_thursday_money_l1846_184689


namespace NUMINAMATH_CALUDE_money_redistribution_l1846_184670

-- Define the initial amounts for each person
variable (a b c d : ℝ)

-- Define the redistribution function
def redistribute (x y z w : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (x - (y + z + w), 2*y, 2*z, 2*w)

-- Theorem statement
theorem money_redistribution (h1 : c = 24) :
  let (a', b', c', d') := redistribute a b c d
  let (a'', b'', c'', d'') := redistribute b' a' c' d'
  let (a''', b''', c''', d''') := redistribute c'' a'' b'' d''
  let (a_final, b_final, c_final, d_final) := redistribute d''' a''' b''' c'''
  c_final = c → a + b + c + d = 96 := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_l1846_184670


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l1846_184692

def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

theorem f_inequality_a_range :
  {a : ℝ | ∃ x, f x ≤ 3/2 * a^2 - a} = {a : ℝ | a ≥ 5/3 ∨ a ≤ -1} :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l1846_184692


namespace NUMINAMATH_CALUDE_special_number_property_l1846_184690

theorem special_number_property (X : ℕ) : 
  (3 + X % 26 = X / 26) ∧ (X % 29 = X / 29) → X = 270 ∨ X = 540 := by
  sorry

end NUMINAMATH_CALUDE_special_number_property_l1846_184690


namespace NUMINAMATH_CALUDE_central_angle_of_chord_l1846_184639

theorem central_angle_of_chord (α : Real) (chord_length : Real) :
  (∀ R, R = 1 → chord_length = Real.sqrt 3 → 2 * Real.sin (α / 2) = chord_length) →
  α = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_of_chord_l1846_184639


namespace NUMINAMATH_CALUDE_books_remaining_pauls_remaining_books_l1846_184691

theorem books_remaining (initial books_given books_sold : ℕ) :
  initial ≥ books_given + books_sold →
  initial - (books_given + books_sold) = initial - books_given - books_sold :=
by
  sorry

theorem pauls_remaining_books :
  134 - (39 + 27) = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_books_remaining_pauls_remaining_books_l1846_184691


namespace NUMINAMATH_CALUDE_product_of_binomials_l1846_184697

theorem product_of_binomials (a : ℝ) : (a + 2) * (2 * a - 3) = 2 * a^2 + a - 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binomials_l1846_184697


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l1846_184614

/-- Configuration of five spheres with a tetrahedron -/
structure SpheresTetrahedron where
  /-- Radius of each sphere -/
  radius : ℝ
  /-- Distance between centers of adjacent spheres on the square -/
  square_side : ℝ
  /-- Height of the top sphere's center above the square -/
  height : ℝ
  /-- Edge length of the tetrahedron -/
  tetra_edge : ℝ
  /-- The radius is 2 -/
  radius_eq : radius = 2
  /-- The square side is twice the diameter -/
  square_side_eq : square_side = 4 * radius
  /-- The height is equal to the diameter -/
  height_eq : height = 2 * radius
  /-- The tetrahedron edge is the distance from a lower sphere to the top sphere -/
  tetra_edge_eq : tetra_edge ^ 2 = square_side ^ 2 + height ^ 2

/-- Theorem: The edge length of the tetrahedron is 4√2 -/
theorem tetrahedron_edge_length (config : SpheresTetrahedron) : 
  config.tetra_edge = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l1846_184614


namespace NUMINAMATH_CALUDE_unique_solution_prime_power_equation_l1846_184629

theorem unique_solution_prime_power_equation :
  ∀ (p q : ℕ) (n m : ℕ),
    Prime p → Prime q → n ≥ 2 → m ≥ 2 →
    (p^n = q^m + 1 ∨ p^n = q^m - 1) →
    (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_prime_power_equation_l1846_184629


namespace NUMINAMATH_CALUDE_apples_left_theorem_l1846_184616

def num_baskets : ℕ := 11
def num_children : ℕ := 10
def initial_apples : ℕ := 1000

def apples_picked (basket : ℕ) : ℕ := basket * num_children

def total_apples_picked : ℕ := (List.range num_baskets).map (λ i => apples_picked (i + 1)) |>.sum

theorem apples_left_theorem :
  initial_apples - total_apples_picked = 340 := by sorry

end NUMINAMATH_CALUDE_apples_left_theorem_l1846_184616


namespace NUMINAMATH_CALUDE_two_y_squared_over_x_is_fraction_l1846_184646

/-- A fraction is an expression with a variable in the denominator -/
def is_fraction (numerator denominator : ℚ) : Prop :=
  ∃ (x : ℚ), denominator = x

/-- The expression 2y^2/x is a fraction -/
theorem two_y_squared_over_x_is_fraction (x y : ℚ) :
  is_fraction (2 * y^2) x :=
sorry

end NUMINAMATH_CALUDE_two_y_squared_over_x_is_fraction_l1846_184646


namespace NUMINAMATH_CALUDE_modulus_of_2_plus_i_l1846_184622

/-- The modulus of the complex number 2 + i is √5 -/
theorem modulus_of_2_plus_i : Complex.abs (2 + Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_plus_i_l1846_184622


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1846_184633

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, x) acc => acc + if x then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldr (fun (i, x) acc => acc + x * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]
  let ternary_num := [2, 1, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 299 := by
sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l1846_184633


namespace NUMINAMATH_CALUDE_original_number_l1846_184698

theorem original_number (y : ℚ) : (1 - (1 / y) = 5 / 4) → y = -4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1846_184698


namespace NUMINAMATH_CALUDE_gcd_triple_existence_l1846_184645

theorem gcd_triple_existence (S : Set ℕ+) (hS_infinite : Set.Infinite S)
  (a b c d : ℕ+) (hab : a ∈ S) (hbc : b ∈ S) (hcd : c ∈ S) (hda : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (hgcd : Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ x y z : ℕ+, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_triple_existence_l1846_184645


namespace NUMINAMATH_CALUDE_shaded_area_in_circle_l1846_184649

/-- The area of a specific shaded region in a circle -/
theorem shaded_area_in_circle (r : ℝ) (h : r = 5) :
  let circle_area := π * r^2
  let triangle_area := r^2 / 2
  let sector_area := circle_area / 4
  2 * triangle_area + 2 * sector_area = 25 + 25 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_circle_l1846_184649


namespace NUMINAMATH_CALUDE_plumber_pipe_cost_l1846_184632

/-- The total cost of pipes bought by a plumber -/
def total_cost (copper_length plastic_length price_per_meter : ℕ) : ℕ :=
  (copper_length + plastic_length) * price_per_meter

/-- Theorem stating the total cost for the plumber's purchase -/
theorem plumber_pipe_cost :
  let copper_length : ℕ := 10
  let plastic_length : ℕ := copper_length + 5
  let price_per_meter : ℕ := 4
  total_cost copper_length plastic_length price_per_meter = 100 := by
  sorry

end NUMINAMATH_CALUDE_plumber_pipe_cost_l1846_184632


namespace NUMINAMATH_CALUDE_combined_solid_volume_l1846_184686

/-- The volume of the combined solid with a square base and triangular prism on top -/
theorem combined_solid_volume (s : ℝ) (h : s = 8 * Real.sqrt 2) :
  let original_volume := (Real.sqrt 2 * (2 * s)^3) / 24
  let prism_volume := (s^3 * Real.sqrt 15) / 4
  original_volume + prism_volume = 2048 + 576 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_combined_solid_volume_l1846_184686


namespace NUMINAMATH_CALUDE_all_students_same_classroom_l1846_184618

/-- The probability that all three students choose the same classroom when randomly selecting between two classrooms with equal probability. -/
theorem all_students_same_classroom (num_classrooms : ℕ) (num_students : ℕ) : 
  num_classrooms = 2 → num_students = 3 → (1 : ℚ) / 4 = 
    (1 : ℚ) / num_classrooms^num_students + (1 : ℚ) / num_classrooms^num_students :=
by sorry

end NUMINAMATH_CALUDE_all_students_same_classroom_l1846_184618


namespace NUMINAMATH_CALUDE_system_solution_l1846_184643

theorem system_solution (a b c : ℝ) 
  (eq1 : b + c = 15 - 2*a)
  (eq2 : a + c = -10 - 4*b)
  (eq3 : a + b = 8 - 2*c) :
  3*a + 3*b + 3*c = 39/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1846_184643


namespace NUMINAMATH_CALUDE_min_chord_length_l1846_184625

def circle_center : ℝ × ℝ := (3, 2)
def circle_radius : ℝ := 3
def point : ℝ × ℝ := (1, 1)

theorem min_chord_length :
  let d := Real.sqrt ((circle_center.1 - point.1)^2 + (circle_center.2 - point.2)^2)
  2 * Real.sqrt (circle_radius^2 - d^2) = 4 := by sorry

end NUMINAMATH_CALUDE_min_chord_length_l1846_184625


namespace NUMINAMATH_CALUDE_picture_area_theorem_l1846_184635

theorem picture_area_theorem (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : 2*x*y + 9*x + 4*y + 12 = 60) : 
  x * y = 15 := by
sorry

end NUMINAMATH_CALUDE_picture_area_theorem_l1846_184635


namespace NUMINAMATH_CALUDE_dining_bill_share_l1846_184615

/-- Given a total bill, number of people, and tip percentage, calculate the amount each person should pay. -/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  let total_with_tip := total_bill * (1 + tip_percentage)
  total_with_tip / num_people

/-- Prove that for a bill of $139.00 split among 5 people with a 10% tip, each person should pay $30.58. -/
theorem dining_bill_share :
  calculate_share 139 5 (1/10) = 3058/100 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l1846_184615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1846_184624

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1846_184624


namespace NUMINAMATH_CALUDE_prime_sum_of_composites_l1846_184634

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem prime_sum_of_composites :
  (∃ p : ℕ, Nat.Prime p ∧ p = 13 ∧ 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) ∧
  (∀ p : ℕ, Nat.Prime p → p > 13 → 
    ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_of_composites_l1846_184634


namespace NUMINAMATH_CALUDE_f_at_2023_half_l1846_184621

/-- A function that is odd and symmetric about x = 1 -/
def f (x : ℝ) : ℝ :=
  sorry

/-- The function f is odd -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The function f is symmetric about x = 1 -/
axiom f_sym (x : ℝ) : f x = f (2 - x)

/-- The function f is defined as 2^x + b for x ∈ [0,1] -/
axiom f_def (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 2^x + Real.pi

/-- The main theorem -/
theorem f_at_2023_half : f (2023/2) = 1 - Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_f_at_2023_half_l1846_184621


namespace NUMINAMATH_CALUDE_simplify_expression_l1846_184669

theorem simplify_expression : (5 * 10^9) / (2 * 10^5 * 5) = 5000 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1846_184669


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1846_184674

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference :
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1846_184674


namespace NUMINAMATH_CALUDE_cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l1846_184601

/-- Definition of Ω(t) function -/
def is_omega_t_function (f : ℝ → ℝ) (t a b : ℝ) : Prop :=
  a < b ∧ t > 0 ∧
  ((∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y)) ∧
  ((∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≤ f y) ∨ (∀ x y, a + t ≤ x ∧ x < y ∧ y ≤ b + t → f x ≥ f y))

/-- Theorem: cos x is an Ω(2) function on [0,1] -/
theorem cos_is_omega_2_on_0_1 : is_omega_t_function Real.cos 2 0 1 := by sorry

/-- Theorem: Characterization of t for sin x to be an Ω(t) function on [-π/2, π/2] -/
theorem sin_omega_t_characterization (t : ℝ) : 
  is_omega_t_function Real.sin t (-π/2) (π/2) ↔ ∃ k : ℤ, t = 2 * π * k ∧ k > 0 := by sorry

/-- Theorem: Range of sin α + sin β for Ω functions -/
theorem sin_sum_range_for_omega_functions (α β : ℝ) :
  (∃ a B, is_omega_t_function Real.sin β a (α + B) ∧ is_omega_t_function Real.sin α B (α + β)) →
  (0 < Real.sin α + Real.sin β ∧ Real.sin α + Real.sin β ≤ 1) ∨ Real.sin α + Real.sin β = 2 := by sorry

end NUMINAMATH_CALUDE_cos_is_omega_2_on_0_1_sin_omega_t_characterization_sin_sum_range_for_omega_functions_l1846_184601


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l1846_184668

theorem opposite_of_one_half : -(1/2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l1846_184668


namespace NUMINAMATH_CALUDE_thomas_final_amount_l1846_184631

-- Define the initial amounts
def michael_initial : ℚ := 42
def thomas_initial : ℚ := 17

-- Define the percentages
def michael_give_percent : ℚ := 35 / 100
def thomas_book_percent : ℚ := 25 / 100

-- Define the candy expense
def candy_expense : ℚ := 5

-- Theorem statement
theorem thomas_final_amount :
  let michael_give := michael_initial * michael_give_percent
  let thomas_after_michael := thomas_initial + michael_give
  let thomas_after_candy := thomas_after_michael - candy_expense
  let book_expense := thomas_after_candy * thomas_book_percent
  let thomas_final := thomas_after_candy - book_expense
  thomas_final = 20.02 := by sorry

end NUMINAMATH_CALUDE_thomas_final_amount_l1846_184631


namespace NUMINAMATH_CALUDE_median_inequality_l1846_184650

-- Define a triangle with sides a, b, c and medians sa, sb, sc
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sa : ℝ
  sb : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- State the theorem
theorem median_inequality (t : Triangle) :
  (t.sa^2 / (t.b * t.c)) + (t.sb^2 / (t.c * t.a)) + (t.sc^2 / (t.a * t.b)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l1846_184650


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l1846_184637

theorem negative_sixty_four_to_seven_thirds :
  (-64 : ℝ) ^ (7/3) = -1024 := by sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l1846_184637


namespace NUMINAMATH_CALUDE_vector_collinearity_l1846_184608

/-- Given vectors a, b, and c in ℝ², prove that if (a + 2b) is collinear with (3a - c),
    then the y-component of b equals -79/14. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h : a = (2, -3) ∧ b.1 = 4 ∧ c = (-1, 1)) :
  (∃ (k : ℝ), k • (a + 2 • b) = 3 • a - c) → b.2 = -79/14 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1846_184608


namespace NUMINAMATH_CALUDE_rock_paper_scissors_lizard_spock_probability_l1846_184602

theorem rock_paper_scissors_lizard_spock_probability :
  let num_players : ℕ := 3
  let num_options : ℕ := 5
  let options_defeated_per_choice : ℕ := 2

  let prob_one_choice_defeats_another : ℚ := options_defeated_per_choice / num_options
  let prob_one_player_defeats_both_others : ℚ := prob_one_choice_defeats_another ^ 2
  let total_probability : ℚ := num_players * prob_one_player_defeats_both_others

  total_probability = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_lizard_spock_probability_l1846_184602


namespace NUMINAMATH_CALUDE_essay_word_count_l1846_184600

theorem essay_word_count 
  (intro_length : ℕ) 
  (body_section_length : ℕ) 
  (num_body_sections : ℕ) 
  (h1 : intro_length = 450)
  (h2 : num_body_sections = 4)
  (h3 : body_section_length = 800) : 
  intro_length + 3 * intro_length + num_body_sections * body_section_length = 5000 :=
by sorry

end NUMINAMATH_CALUDE_essay_word_count_l1846_184600


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1846_184683

/-- Given a sum P put at simple interest rate R% for 1 year, 
    if increasing the rate by 6% results in Rs. 30 more interest, 
    then P = 500. -/
theorem simple_interest_problem (P R : ℝ) 
  (h1 : P * (R + 6) / 100 = P * R / 100 + 30) : 
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1846_184683


namespace NUMINAMATH_CALUDE_greatest_multiple_of_6_and_5_less_than_1000_l1846_184604

theorem greatest_multiple_of_6_and_5_less_than_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  n < 1000 ∧ 
  ∀ m : ℕ, (6 ∣ m ∧ 5 ∣ m ∧ m < 1000) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_6_and_5_less_than_1000_l1846_184604


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1846_184627

/-- A line passing through point (2,1) and perpendicular to x-2y+1=0 has equation 2x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ p : ℝ × ℝ, p ∈ l ↔ 2 * p.1 + p.2 - 5 = 0) → -- l is defined by 2x + y - 5 = 0
    ((2, 1) ∈ l) →  -- l passes through (2,1)
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → 
      (p.1 - q.1) * (1 - 2) + (p.2 - q.2) * (1 - (-1/2)) = 0) → -- l is perpendicular to x-2y+1=0
    True := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1846_184627


namespace NUMINAMATH_CALUDE_parents_age_when_mark_born_l1846_184661

/-- Given the ages of Mark and John, and their relation to their parents' age, 
    proves the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born (mark_age john_age parents_age : ℕ) : 
  mark_age = 18 →
  john_age = mark_age - 10 →
  parents_age = 5 * john_age →
  parents_age - mark_age = 22 :=
by sorry

end NUMINAMATH_CALUDE_parents_age_when_mark_born_l1846_184661


namespace NUMINAMATH_CALUDE_maggie_total_spent_l1846_184613

def plant_books : ℕ := 20
def fish_books : ℕ := 7
def magazines : ℕ := 25
def book_cost : ℕ := 25
def magazine_cost : ℕ := 5

theorem maggie_total_spent : 
  (plant_books + fish_books) * book_cost + magazines * magazine_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_maggie_total_spent_l1846_184613


namespace NUMINAMATH_CALUDE_cylindrical_bucket_height_l1846_184684

/-- The height of a cylindrical bucket given its radius and the dimensions of a conical heap formed when emptied -/
theorem cylindrical_bucket_height (r_cylinder r_cone h_cone : ℝ) (h_cylinder : ℝ) : 
  r_cylinder = 21 →
  r_cone = 63 →
  h_cone = 12 →
  r_cylinder^2 * h_cylinder = (1/3) * r_cone^2 * h_cone →
  h_cylinder = 36 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_bucket_height_l1846_184684


namespace NUMINAMATH_CALUDE_intersection_line_and_chord_length_l1846_184678

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line
def L (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem intersection_line_and_chord_length :
  -- The line L passes through the intersection points of C₁ and C₂
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → L x y) ∧
  -- The length of the common chord is √2
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    C₁ x₁ y₁ ∧ C₂ x₁ y₁ ∧
    C₁ x₂ y₂ ∧ C₂ x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_and_chord_length_l1846_184678


namespace NUMINAMATH_CALUDE_summer_camp_total_l1846_184665

theorem summer_camp_total (T : ℕ) : 
  (T / 2 : ℕ) = T - T / 2 →  -- Half of kids go to soccer camp
  (T / 2) / 3 + 750 = T / 2 →  -- 1/3 morning + 750 afternoon = total soccer
  ((T - T / 2) / 5) / 3 = 200 →  -- 1/3 of 1/5 of other half go to morning basketball
  T = 6000 := by
  sorry

end NUMINAMATH_CALUDE_summer_camp_total_l1846_184665


namespace NUMINAMATH_CALUDE_min_value_expression_l1846_184612

theorem min_value_expression :
  ∃ (s₀ t₀ : ℝ), ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ 2 ∧
  (s₀ + 5 - 3 * |Real.cos t₀|)^2 + (s₀ - 2 * |Real.sin t₀|)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1846_184612


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1846_184609

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 3*i) / (2 + 5*i) = (-13 : ℝ) / 29 - (11 : ℝ) / 29 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1846_184609


namespace NUMINAMATH_CALUDE_segments_complete_circle_num_segments_minimal_l1846_184641

/-- The number of equal segments that can be drawn around a circle,
    where each segment subtends an arc of 120°. -/
def num_segments : ℕ := 3

/-- The measure of the arc subtended by each segment in degrees. -/
def arc_measure : ℕ := 120

/-- Theorem stating that the number of segments multiplied by the arc measure
    equals a full circle (360°). -/
theorem segments_complete_circle :
  num_segments * arc_measure = 360 := by sorry

/-- Theorem stating that num_segments is the smallest positive integer
    that satisfies the segments_complete_circle property. -/
theorem num_segments_minimal :
  ∀ n : ℕ, 0 < n → n * arc_measure = 360 → num_segments ≤ n := by sorry

end NUMINAMATH_CALUDE_segments_complete_circle_num_segments_minimal_l1846_184641


namespace NUMINAMATH_CALUDE_dislike_both_count_l1846_184644

/-- The number of people who don't like both radio and music in a poll -/
def people_dislike_both (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ) : ℕ :=
  ⌊(radio_dislike_percent * music_dislike_percent * total : ℚ)⌋₊

/-- Theorem about the number of people who don't like both radio and music -/
theorem dislike_both_count :
  people_dislike_both 1500 (35/100) (15/100) = 79 := by
  sorry

#eval people_dislike_both 1500 (35/100) (15/100)

end NUMINAMATH_CALUDE_dislike_both_count_l1846_184644


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_minus_one_l1846_184675

theorem largest_two_digit_multiple_minus_one : ∃ (n : ℕ), n = 83 ∧ 
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ 
    (∃ k : ℕ, m + 1 = 3 * k) ∧ 
    (∃ k : ℕ, m + 1 = 4 * k) ∧ 
    (∃ k : ℕ, m + 1 = 5 * k) ∧ 
    (∃ k : ℕ, m + 1 = 7 * k) → 
  m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_minus_one_l1846_184675


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l1846_184605

/-- The area of a square containing four circles, each with a radius of 10 inches
    and touching two sides of the square and two other circles. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 10) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 1600 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l1846_184605


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1846_184655

/-- An inverse proportion function passing through (3, -5) is in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ k : ℝ,
  (∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧ f 3 = -5) →
  (∀ x y : ℝ, x ≠ 0 ∧ y = k / x → (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry


end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l1846_184655


namespace NUMINAMATH_CALUDE_bus_cost_proof_l1846_184606

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := 9.85

theorem bus_cost_proof : bus_cost = 1.50 := by sorry

end NUMINAMATH_CALUDE_bus_cost_proof_l1846_184606


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1846_184688

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 65 = 0) ∧ (a * 5^2 + 3 * 5 - 65 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1846_184688


namespace NUMINAMATH_CALUDE_hancho_milk_consumption_l1846_184658

theorem hancho_milk_consumption (total_milk : Real) (yeseul_milk : Real) (gayoung_extra : Real) (remaining_milk : Real) :
  total_milk = 1 →
  yeseul_milk = 0.1 →
  gayoung_extra = 0.2 →
  remaining_milk = 0.3 →
  total_milk - yeseul_milk - (yeseul_milk + gayoung_extra) - remaining_milk = 0.3 := by
  sorry

#check hancho_milk_consumption

end NUMINAMATH_CALUDE_hancho_milk_consumption_l1846_184658


namespace NUMINAMATH_CALUDE_product_of_negative_real_part_solutions_l1846_184671

theorem product_of_negative_real_part_solutions :
  let solutions : List (ℂ) := [2 * (Complex.exp (Complex.I * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 3 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 5 * Real.pi / 4)),
                               2 * (Complex.exp (Complex.I * 7 * Real.pi / 4))]
  let negative_real_part_solutions := solutions.filter (fun z => z.re < 0)
  ∀ z ∈ solutions, z^4 = -16 →
  negative_real_part_solutions.prod = 4 := by
sorry

end NUMINAMATH_CALUDE_product_of_negative_real_part_solutions_l1846_184671


namespace NUMINAMATH_CALUDE_amy_music_files_l1846_184603

theorem amy_music_files :
  ∀ (initial_music_files initial_video_files deleted_files remaining_files : ℕ),
    initial_video_files = 21 →
    deleted_files = 23 →
    remaining_files = 2 →
    initial_music_files + initial_video_files - deleted_files = remaining_files →
    initial_music_files = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_music_files_l1846_184603


namespace NUMINAMATH_CALUDE_minimize_S_l1846_184672

/-- The sum of squared differences function -/
def S (x y z : ℝ) : ℝ :=
  (x + y + z - 10)^2 + (x + y - z - 7)^2 + (x - y + z - 6)^2 + (-x + y + z - 5)^2

/-- Theorem stating that (4.5, 4, 3.5) minimizes S -/
theorem minimize_S :
  ∀ x y z : ℝ, S x y z ≥ S 4.5 4 3.5 := by sorry

end NUMINAMATH_CALUDE_minimize_S_l1846_184672


namespace NUMINAMATH_CALUDE_equation_solution_l1846_184682

-- Define the operation "*"
def star_op (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem equation_solution :
  ∃! x : ℝ, star_op (x - 4) 1 = 0 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1846_184682


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1846_184699

theorem sqrt_sum_equality (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1846_184699


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1846_184676

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x^2 - 5 * x + 2 > 0) ↔ (x < (1 : ℝ) / 2 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1846_184676


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1846_184638

theorem yellow_marbles_count (total : ℕ) (yellow green red blue : ℕ) : 
  total = 60 →
  green = yellow / 2 →
  red = blue →
  blue = total / 4 →
  total = yellow + green + red + blue →
  yellow = 20 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1846_184638


namespace NUMINAMATH_CALUDE_calculation_proof_l1846_184611

theorem calculation_proof :
  (5.42 - (3.75 - 0.58) = 2.25) ∧
  ((4/5) * 7.7 + 0.8 * 3.3 - (4/5) = 8) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1846_184611


namespace NUMINAMATH_CALUDE_dye_job_price_is_correct_l1846_184620

/-- The price of a haircut in dollars -/
def haircut_price : ℕ := 30

/-- The price of a perm in dollars -/
def perm_price : ℕ := 40

/-- The cost of hair dye for one dye job in dollars -/
def dye_cost : ℕ := 10

/-- The number of haircuts scheduled -/
def num_haircuts : ℕ := 4

/-- The number of perms scheduled -/
def num_perms : ℕ := 1

/-- The number of dye jobs scheduled -/
def num_dye_jobs : ℕ := 2

/-- The amount of tips in dollars -/
def tips : ℕ := 50

/-- The total earnings at the end of the day in dollars -/
def total_earnings : ℕ := 310

/-- The price of a dye job in dollars -/
def dye_job_price : ℕ := 60

theorem dye_job_price_is_correct : 
  num_haircuts * haircut_price + 
  num_perms * perm_price + 
  num_dye_jobs * (dye_job_price - dye_cost) + 
  tips = total_earnings := by sorry

end NUMINAMATH_CALUDE_dye_job_price_is_correct_l1846_184620


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1846_184687

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 = 5 * x - 1) →
    (∀ x, a * x^2 + b * x + c = 0) →
    (∀ x, 3 * x^2 - 5 * x + 1 = 0) →
    a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1846_184687


namespace NUMINAMATH_CALUDE_pentagon_area_bound_l1846_184610

-- Define the pentagon ABCDE
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_convex_pentagon (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def angle (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def area (A B C D E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem pentagon_area_bound 
  (h_convex : is_convex_pentagon A B C D E)
  (h_angle_EAB : angle E A B = 2 * π / 3)
  (h_angle_ABC : angle A B C = 2 * π / 3)
  (h_angle_ADB : angle A D B = π / 6)
  (h_angle_CDE : angle C D E = π / 3)
  (h_side_AB : distance A B = 1) :
  area A B C D E < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_bound_l1846_184610


namespace NUMINAMATH_CALUDE_company_salary_theorem_l1846_184626

/-- Proves that given a company with 15 managers earning an average of $90,000 and 75 associates,
    if the company's overall average salary is $40,000, then the average salary of associates is $30,000. -/
theorem company_salary_theorem (num_managers : ℕ) (num_associates : ℕ) 
    (avg_salary_managers : ℝ) (avg_salary_company : ℝ) : 
    num_managers = 15 →
    num_associates = 75 →
    avg_salary_managers = 90000 →
    avg_salary_company = 40000 →
    (num_managers * avg_salary_managers + num_associates * (30000 : ℝ)) / (num_managers + num_associates) = avg_salary_company :=
by sorry

end NUMINAMATH_CALUDE_company_salary_theorem_l1846_184626


namespace NUMINAMATH_CALUDE_range_of_m_l1846_184617

/-- The line x - 2y + 3 = 0 and the parabola y² = mx (m ≠ 0) have no points of intersection -/
def p (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- The equation x²/(5-2m) + y²/m = 1 represents a hyperbola -/
def q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2/(5-2*m) + y^2/m = 1 ∧ m*(5-2*m) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) →
    m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1846_184617


namespace NUMINAMATH_CALUDE_cubic_root_implies_h_value_l1846_184652

theorem cubic_root_implies_h_value :
  ∀ h : ℝ, (3 : ℝ)^3 + h * 3 - 20 = 0 → h = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_h_value_l1846_184652


namespace NUMINAMATH_CALUDE_local_value_of_four_l1846_184628

/-- The local value of a digit in a number. -/
def local_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- The sum of local values of all digits in 2345. -/
def total_sum : ℕ := 2345

/-- The local values of digits 2, 3, and 5 in 2345. -/
def known_values : ℕ := local_value 2 3 + local_value 3 2 + local_value 5 0

/-- The local value of the remaining digit (4) in 2345. -/
def remaining_value : ℕ := total_sum - known_values

theorem local_value_of_four :
  remaining_value = local_value 4 1 :=
sorry

end NUMINAMATH_CALUDE_local_value_of_four_l1846_184628


namespace NUMINAMATH_CALUDE_total_highlighters_l1846_184662

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 6) (h2 : yellow = 2) (h3 : blue = 4) :
  pink + yellow + blue = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l1846_184662


namespace NUMINAMATH_CALUDE_parallelogram_area_l1846_184680

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 6), and (5, 6) is 24 square units. -/
theorem parallelogram_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (5, 6)
  let D : ℝ × ℝ := (1, 6)
  let area := abs ((B.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (B.2 - A.2))
  area = 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1846_184680


namespace NUMINAMATH_CALUDE_point_c_coordinates_l1846_184693

/-- Given points A and B in ℝ², and a relationship between vectors AC and CB,
    prove that point C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (3, 0) → 
  C - A = -2 • (B - C) → 
  C = (4, -3) := by
sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l1846_184693


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1846_184667

/-- Given a boat's travel times for the same distance upstream and downstream, 
    calculate its speed in still water. -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (time_downstream time_upstream : ℝ) 
  (h_distance_positive : distance > 0)
  (h_time_downstream_positive : time_downstream > 0)
  (h_time_upstream_positive : time_upstream > 0)
  (h_downstream : distance / time_downstream = 10)
  (h_upstream : distance / time_upstream = 4) : 
  ∃ (boat_speed : ℝ), boat_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1846_184667


namespace NUMINAMATH_CALUDE_ninth_term_value_l1846_184647

/-- An arithmetic sequence with specified third and sixth terms -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  third_term : a + 2 * d = 25
  sixth_term : a + 5 * d = 31

/-- The ninth term of the arithmetic sequence -/
def ninth_term (seq : ArithmeticSequence) : ℝ := seq.a + 8 * seq.d

theorem ninth_term_value (seq : ArithmeticSequence) : ninth_term seq = 37 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l1846_184647


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1846_184648

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1846_184648


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1846_184640

/-- Given a quadratic function y = ax^2 + bx + 2 passing through (-1, 0), 
    prove that 2a - 2b = -4 -/
theorem quadratic_function_property (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + 2) → 
  (0 = a * (-1)^2 + b * (-1) + 2) → 
  2 * a - 2 * b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1846_184640


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l1846_184642

/-- Given two squares C and D, where C has a perimeter of 32 cm and D has an area that is half the area of C,
    prove that the perimeter of D is 16√2 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- Perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 2) →  -- Area of D is half the area of C
  (4 * D = 16 * Real.sqrt 2) :=  -- Perimeter of D is 16√2 cm
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l1846_184642


namespace NUMINAMATH_CALUDE_initial_pens_l1846_184679

def double_weekly (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * double_weekly initial n

theorem initial_pens (initial : ℕ) :
  double_weekly initial 4 = 32 → initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_l1846_184679


namespace NUMINAMATH_CALUDE_tinas_earnings_l1846_184654

/-- Calculates the total earnings for a worker given their hourly rate, hours worked per day, 
    number of days worked, and regular hours per day before overtime. -/
def calculate_earnings (hourly_rate : ℚ) (hours_per_day : ℕ) (days_worked : ℕ) (regular_hours : ℕ) : ℚ :=
  let regular_pay := hourly_rate * regular_hours * days_worked
  let overtime_hours := if hours_per_day > regular_hours then hours_per_day - regular_hours else 0
  let overtime_rate := hourly_rate * (1 + 1/2)
  let overtime_pay := overtime_rate * overtime_hours * days_worked
  regular_pay + overtime_pay

/-- Theorem stating that Tina's earnings for 5 days of work at 10 hours per day 
    with an $18.00 hourly rate is $990.00. -/
theorem tinas_earnings : 
  calculate_earnings 18 10 5 8 = 990 := by
  sorry

end NUMINAMATH_CALUDE_tinas_earnings_l1846_184654


namespace NUMINAMATH_CALUDE_coordinates_of_P_wrt_x_axis_l1846_184656

/-- Given a point P in the Cartesian coordinate system, this function
    returns its coordinates with respect to the x-axis. -/
def coordinates_wrt_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem stating that the coordinates of P(-2, 3) with respect to the x-axis are (-2, -3). -/
theorem coordinates_of_P_wrt_x_axis :
  coordinates_wrt_x_axis (-2, 3) = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_P_wrt_x_axis_l1846_184656


namespace NUMINAMATH_CALUDE_corveus_sleep_hours_l1846_184653

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the doctor's recommended hours of sleep per day -/
def recommended_sleep_per_day : ℕ := 6

/-- Represents the sleep deficit in hours per week -/
def sleep_deficit_per_week : ℕ := 14

/-- Calculates Corveus's actual sleep hours per day -/
def actual_sleep_per_day : ℚ :=
  (recommended_sleep_per_day * days_in_week - sleep_deficit_per_week) / days_in_week

/-- Proves that Corveus sleeps 4 hours per day given the conditions -/
theorem corveus_sleep_hours :
  actual_sleep_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_corveus_sleep_hours_l1846_184653


namespace NUMINAMATH_CALUDE_therapy_pricing_theorem_l1846_184623

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price for the first hour
  additional_hour : ℕ  -- Price for each additional hour
  total_5_hours : ℕ  -- Total charge for 5 hours of therapy

/-- Given the pricing structure, calculates the total charge for 2 hours of therapy. -/
def charge_for_2_hours (pricing : TherapyPricing) : ℕ :=
  pricing.first_hour + pricing.additional_hour

/-- Theorem stating the relationship between the pricing structure and the charge for 2 hours. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 40)
  (h2 : pricing.total_5_hours = 375)
  (h3 : pricing.first_hour + 4 * pricing.additional_hour = pricing.total_5_hours) :
  charge_for_2_hours pricing = 174 := by
  sorry

#eval charge_for_2_hours { first_hour := 107, additional_hour := 67, total_5_hours := 375 }

end NUMINAMATH_CALUDE_therapy_pricing_theorem_l1846_184623


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_odds_l1846_184681

theorem largest_of_four_consecutive_odds (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →  -- average is 24
  (x + 6 = 27) :=                         -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_odds_l1846_184681


namespace NUMINAMATH_CALUDE_average_age_increase_l1846_184685

theorem average_age_increase 
  (n : Nat) 
  (initial_avg : ℝ) 
  (man1_age man2_age : ℝ) 
  (women_avg : ℝ) : 
  n = 8 → 
  man1_age = 20 → 
  man2_age = 22 → 
  women_avg = 29 → 
  ((n * initial_avg - man1_age - man2_age + 2 * women_avg) / n) - initial_avg = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1846_184685


namespace NUMINAMATH_CALUDE_deck_problem_l1846_184673

theorem deck_problem (r b : ℕ) : 
  r / (r + b) = 1 / 5 →
  r / (r + (b + 6)) = 1 / 7 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_deck_problem_l1846_184673


namespace NUMINAMATH_CALUDE_temperature_difference_l1846_184666

/-- The difference between the average highest and lowest temperatures in Shangri-La -/
theorem temperature_difference (avg_high avg_low : ℝ) 
  (h1 : avg_high = 9)
  (h2 : avg_low = -5) :
  avg_high - avg_low = 14 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1846_184666


namespace NUMINAMATH_CALUDE_sarahs_age_l1846_184636

theorem sarahs_age (game_formula : ℕ → ℕ) (name_letters : ℕ) (marriage_age : ℕ) :
  game_formula name_letters = marriage_age →
  name_letters = 5 →
  marriage_age = 23 →
  ∃ current_age : ℕ, game_formula 5 = 23 ∧ current_age = 9 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_age_l1846_184636


namespace NUMINAMATH_CALUDE_symmetrical_letters_count_l1846_184695

-- Define a function to check if a character is symmetrical
def is_symmetrical (c : Char) : Bool :=
  c = 'A' ∨ c = 'H' ∨ c = 'I' ∨ c = 'M' ∨ c = 'O' ∨ c = 'T' ∨ c = 'U' ∨ c = 'V' ∨ c = 'W' ∨ c = 'X' ∨ c = 'Y'

-- Define the sign text
def sign_text : String := "PUNK CD FOR SALE"

-- Theorem statement
theorem symmetrical_letters_count :
  (sign_text.toList.filter is_symmetrical).length = 3 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_letters_count_l1846_184695


namespace NUMINAMATH_CALUDE_a_faster_than_b_l1846_184607

/-- Represents a person sawing wood --/
structure Sawyer where
  name : String
  segments_per_piece : ℕ
  total_segments : ℕ

/-- Calculates the number of pieces sawed by a sawyer --/
def pieces_sawed (s : Sawyer) : ℕ := s.total_segments / s.segments_per_piece

/-- Calculates the number of cuts made by a sawyer --/
def cuts_made (s : Sawyer) : ℕ := (s.segments_per_piece - 1) * (pieces_sawed s)

/-- Theorem stating that A takes less time to saw one piece of wood --/
theorem a_faster_than_b (a b : Sawyer)
  (ha : a.name = "A" ∧ a.segments_per_piece = 3 ∧ a.total_segments = 24)
  (hb : b.name = "B" ∧ b.segments_per_piece = 2 ∧ b.total_segments = 28) :
  cuts_made a > cuts_made b := by sorry

end NUMINAMATH_CALUDE_a_faster_than_b_l1846_184607


namespace NUMINAMATH_CALUDE_exponent_division_equality_l1846_184659

theorem exponent_division_equality (a b : ℝ) :
  (a^2 * b)^3 / ((-a * b)^2) = a^4 * b :=
by sorry

end NUMINAMATH_CALUDE_exponent_division_equality_l1846_184659


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l1846_184630

theorem glove_selection_theorem :
  let n : ℕ := 6  -- Total number of glove pairs
  let k : ℕ := 5  -- Number of gloves to select
  let same_pair : ℕ := 2  -- Number of gloves from the same pair

  -- Function to calculate the number of ways to select gloves
  let select_gloves : ℕ :=
    (n.choose 1) *  -- Choose 1 pair for the matching gloves
    ((n - 1).choose (k - same_pair)) *  -- Choose remaining pairs
    (2 ^ (k - same_pair))  -- Select one glove from each remaining pair

  select_gloves = 480 := by sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l1846_184630


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1846_184664

/-- Proves that for a rectangular plot where the length is thrice the breadth
    and the area is 432 sq m, the breadth is 12 m. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
    length = 3 * breadth →
    area = length * breadth →
    area = 432 →
    breadth = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1846_184664


namespace NUMINAMATH_CALUDE_al_original_portion_l1846_184694

/-- Represents the investment scenario with four participants --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- The investment scenario satisfies the given conditions --/
def ValidInvestment (i : Investment) : Prop :=
  i.al + i.betty + i.clare + i.dave = 1200 ∧
  (i.al - 150) + (2 * i.betty) + (2 * i.clare) + (3 * i.dave) = 1800

/-- Theorem stating that Al's original portion was $450 --/
theorem al_original_portion (i : Investment) (h : ValidInvestment i) : i.al = 450 := by
  sorry

#check al_original_portion

end NUMINAMATH_CALUDE_al_original_portion_l1846_184694


namespace NUMINAMATH_CALUDE_complex_number_location_l1846_184696

theorem complex_number_location (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (3 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1846_184696


namespace NUMINAMATH_CALUDE_cuatro_cuinte_equation_l1846_184651

/-- Represents a mapping from letters to digits -/
def LetterToDigit := Char → Nat

/-- Check if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterToDigit) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Convert a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterToDigit) : Nat :=
  s.foldl (fun acc c => 10 * acc + m c) 0

/-- The main theorem to prove -/
theorem cuatro_cuinte_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  170349 + 170349 + 1700349 + 1700349 + 170349 = 3852345 := by
  sorry

/-- Lemma: The mapping satisfies the equation -/
lemma mapping_satisfies_equation (m : LetterToDigit) 
  (h_valid : is_valid_mapping m)
  (h_cuatro : string_to_number "CUATRO" m = 170349)
  (h_cuaatro : string_to_number "CUAATRO" m = 1700349)
  (h_cuinte : string_to_number "CUINTE" m = 3852345) :
  string_to_number "CUATRO" m + string_to_number "CUATRO" m + 
  string_to_number "CUAATRO" m + string_to_number "CUAATRO" m + 
  string_to_number "CUATRO" m = string_to_number "CUINTE" m := by
  sorry

end NUMINAMATH_CALUDE_cuatro_cuinte_equation_l1846_184651


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1846_184657

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (f_prime a (-3) = 0 →
    ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (y = m*x + b ↔ x = 0 ∨ y - f a 0 = m*(x - 0))) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 1 2 → f_prime a x ≤ 0) →
    a ≤ -15/4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1846_184657


namespace NUMINAMATH_CALUDE_S_2n_plus_one_not_div_by_three_l1846_184663

/-- 
For a non-negative integer n, S_n is defined as the sum of squares 
of the coefficients of the polynomial (1+x)^n
-/
def S (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k) ^ 2)

/-- 
For any non-negative integer n, S(2n) + 1 is not divisible by 3
-/
theorem S_2n_plus_one_not_div_by_three (n : ℕ) : ¬ (3 ∣ (S (2 * n) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_S_2n_plus_one_not_div_by_three_l1846_184663


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1846_184677

theorem sum_with_radical_conjugate : 
  let a : ℝ := 15 - Real.sqrt 500
  let radical_conjugate : ℝ := 15 + Real.sqrt 500
  a + radical_conjugate = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1846_184677
