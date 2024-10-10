import Mathlib

namespace calculator_problem_l3168_316854

/-- Represents the possible operations on the calculator --/
inductive Operation
| addOne
| addThree
| double

/-- Applies a single operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.addThree => n + 3
  | Operation.double => n * 2

/-- Applies a sequence of operations to a number --/
def applySequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Checks if a sequence of operations transforms start into target --/
def isValidSequence (start target : ℕ) (ops : List Operation) : Prop :=
  applySequence start ops = target

/-- The main theorem to be proved --/
theorem calculator_problem :
  ∃ (ops : List Operation),
    ops.length = 10 ∧
    isValidSequence 1 410 ops ∧
    ∀ (shorter_ops : List Operation),
      shorter_ops.length < 10 →
      ¬ isValidSequence 1 410 shorter_ops :=
sorry

end calculator_problem_l3168_316854


namespace cards_distribution_l3168_316849

/-- Given 60 cards dealt to 9 people as evenly as possible, 
    the number of people with fewer than 7 cards is 3. -/
theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
    (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 3 := by
  sorry

end cards_distribution_l3168_316849


namespace polygon_sides_count_l3168_316833

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end polygon_sides_count_l3168_316833


namespace joan_seashells_l3168_316831

/-- The number of seashells Joan has after a series of events -/
def final_seashells (initial : ℕ) (given : ℕ) (found : ℕ) (traded : ℕ) (received : ℕ) (lost : ℕ) : ℕ :=
  initial - given + found - traded + received - lost

/-- Theorem stating that Joan ends up with 51 seashells -/
theorem joan_seashells :
  final_seashells 79 63 45 20 15 5 = 51 := by
  sorry

end joan_seashells_l3168_316831


namespace point_C_complex_number_l3168_316850

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem point_C_complex_number (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I := by sorry

end point_C_complex_number_l3168_316850


namespace distance_between_parallel_lines_l3168_316884

/-- The distance between two parallel lines in R² --/
theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ × ℝ := λ t ↦ (4 + 2*t, -1 - 6*t)
  let line2 : ℝ → ℝ × ℝ := λ s ↦ (3 + 2*s, -2 - 6*s)
  let v : ℝ × ℝ := (3 - 4, -2 - (-1))
  let d : ℝ × ℝ := (2, -6)
  let distance := ‖v - (((v.1 * d.1 + v.2 * d.2) / (d.1^2 + d.2^2)) • d)‖
  distance = 2 * Real.sqrt 10 / 5 := by
sorry


end distance_between_parallel_lines_l3168_316884


namespace square_field_dimensions_l3168_316835

/-- Proves that a square field with the given fence properties has a side length of 16000 meters -/
theorem square_field_dimensions (x : ℝ) : 
  x > 0 ∧ 
  (1.6 * x = x^2 / 10000) → 
  x = 16000 := by
sorry

end square_field_dimensions_l3168_316835


namespace new_average_weight_l3168_316803

def initial_average_weight : ℝ := 48
def initial_members : ℕ := 23
def new_person1_weight : ℝ := 78
def new_person2_weight : ℝ := 93

theorem new_average_weight :
  let total_initial_weight := initial_average_weight * initial_members
  let total_new_weight := new_person1_weight + new_person2_weight
  let total_weight := total_initial_weight + total_new_weight
  let new_members := initial_members + 2
  total_weight / new_members = 51 := by
  sorry

end new_average_weight_l3168_316803


namespace arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l3168_316853

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- The number of arrangements with A and B at the ends -/
def arrangements_ends : ℕ := 240

/-- The number of arrangements with A, B, and C together -/
def arrangements_together : ℕ := 720

/-- The number of arrangements with A not at beginning and B not at end -/
def arrangements_not_ends : ℕ := 3720

/-- Theorem for the number of arrangements with A and B at the ends -/
theorem arrangements_ends_correct : 
  arrangements_ends = 2 * Nat.factorial (n - 2) := by sorry

/-- Theorem for the number of arrangements with A, B, and C together -/
theorem arrangements_together_correct : 
  arrangements_together = 6 * Nat.factorial (n - 3) := by sorry

/-- Theorem for the number of arrangements with A not at beginning and B not at end -/
theorem arrangements_not_ends_correct : 
  arrangements_not_ends = Nat.factorial n - 2 * Nat.factorial (n - 1) + Nat.factorial (n - 2) := by sorry

end arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l3168_316853


namespace normal_distribution_std_dev_l3168_316818

theorem normal_distribution_std_dev (μ : ℝ) (x : ℝ) (σ : ℝ) 
  (h1 : μ = 14.5)
  (h2 : x = 11.1)
  (h3 : x = μ - 2 * σ) :
  σ = 1.7 := by
sorry

end normal_distribution_std_dev_l3168_316818


namespace sufficient_condition_l3168_316810

-- Define propositions P and Q
def P (a b c d : ℝ) : Prop := a ≥ b → c > d
def Q (a b e f : ℝ) : Prop := e ≤ f → a < b

-- Main theorem
theorem sufficient_condition (a b c d e f : ℝ) 
  (hP : P a b c d) 
  (hnotQ : ¬(Q a b e f)) : 
  c ≤ d → e ≤ f := by
sorry

end sufficient_condition_l3168_316810


namespace rose_bush_count_l3168_316846

theorem rose_bush_count (initial_bushes planted_bushes : ℕ) :
  initial_bushes = 2 → planted_bushes = 4 →
  initial_bushes + planted_bushes = 6 := by
  sorry

end rose_bush_count_l3168_316846


namespace min_value_x_plus_inverse_equality_condition_l3168_316856

theorem min_value_x_plus_inverse (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
by
  sorry

theorem equality_condition (x : ℝ) (hx : x > 0) : x + 1/x = 2 ↔ x = 1 :=
by
  sorry

end min_value_x_plus_inverse_equality_condition_l3168_316856


namespace decimal_addition_l3168_316888

theorem decimal_addition : 5.763 + 2.489 = 8.152 := by sorry

end decimal_addition_l3168_316888


namespace cube_power_eq_l3168_316830

theorem cube_power_eq : (3^3 * 6^3)^2 = 34062224 := by
  sorry

end cube_power_eq_l3168_316830


namespace two_machines_half_hour_copies_l3168_316851

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the total number of copies made by two machines in a given time. -/
def total_copies (machine1 machine2 : CopyMachine) (minutes : ℕ) : ℕ :=
  (machine1.copies_per_minute + machine2.copies_per_minute) * minutes

/-- Theorem stating that two specific copy machines working together for 30 minutes will produce 2850 copies. -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨40⟩
  let machine2 : CopyMachine := ⟨55⟩
  total_copies machine1 machine2 30 = 2850 := by
  sorry


end two_machines_half_hour_copies_l3168_316851


namespace find_constant_b_l3168_316834

theorem find_constant_b (b d c : ℚ) : 
  (∀ x : ℚ, (7 * x^2 - 5 * x + 11/4) * (d * x^2 + b * x + c) = 
    21 * x^4 - 26 * x^3 + 34 * x^2 - (55/4) * x + 33/4) → 
  b = -11/7 := by
sorry

end find_constant_b_l3168_316834


namespace rope_cutting_l3168_316874

/-- Given two ropes of lengths 18 and 24 meters, this theorem proves that
    the maximum length of equal segments that can be cut from both ropes
    without remainder is 6 meters, and the total number of such segments is 7. -/
theorem rope_cutting (rope1 : ℕ) (rope2 : ℕ) 
  (h1 : rope1 = 18) (h2 : rope2 = 24) : 
  ∃ (segment_length : ℕ) (total_segments : ℕ),
    segment_length = 6 ∧ 
    total_segments = 7 ∧
    rope1 % segment_length = 0 ∧
    rope2 % segment_length = 0 ∧
    rope1 / segment_length + rope2 / segment_length = total_segments ∧
    ∀ (l : ℕ), l > segment_length → (rope1 % l ≠ 0 ∨ rope2 % l ≠ 0) :=
by sorry

end rope_cutting_l3168_316874


namespace triangle_side_angle_inequality_l3168_316807

/-- Triangle inequality for side lengths and angles -/
theorem triangle_side_angle_inequality 
  (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α := by
  sorry

end triangle_side_angle_inequality_l3168_316807


namespace point_outside_circle_l3168_316838

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle_equation (x y : ℝ) := x^2 + y^2 = 24
  ∀ x y, circle_equation x y → (P.1 - x)^2 + (P.2 - y)^2 > 0 :=
by
  sorry

end point_outside_circle_l3168_316838


namespace no_solutions_in_interval_l3168_316857

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Ioo 0 (π / 6) →
  3 * Real.tan (2 * x) - 4 * Real.tan (3 * x) ≠ Real.tan (3 * x) ^ 2 * Real.tan (2 * x) :=
by sorry

end no_solutions_in_interval_l3168_316857


namespace fraction_product_squares_l3168_316828

theorem fraction_product_squares : 
  (4/5)^2 * (3/7)^2 * (2/3)^2 = 64/1225 := by
  sorry

end fraction_product_squares_l3168_316828


namespace mollys_age_l3168_316809

theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / (molly_age : ℚ) = 4 / 3 →
  sandy_age + 6 = 30 →
  molly_age = 18 :=
by sorry

end mollys_age_l3168_316809


namespace least_possible_c_l3168_316891

theorem least_possible_c (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 20 →
  a ≤ b ∧ b ≤ c →
  b - a ≥ 2 ∧ c - b ≥ 2 →
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 →
  b = a + 13 →
  c ≥ 33 ∧ ∀ (c' : ℕ), c' ≥ 33 → c' % 3 = 0 → c' - b ≥ 2 → c ≤ c' :=
by sorry

end least_possible_c_l3168_316891


namespace rectangle_width_l3168_316836

theorem rectangle_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 48 → 
  width = 4 := by
sorry

end rectangle_width_l3168_316836


namespace remainder_of_55_power_55_plus_55_mod_56_l3168_316867

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by
  sorry

end remainder_of_55_power_55_plus_55_mod_56_l3168_316867


namespace current_speed_l3168_316869

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 := by
  sorry

end current_speed_l3168_316869


namespace shorts_cost_l3168_316892

def total_spent : ℝ := 33.56
def shirt_cost : ℝ := 12.14
def jacket_cost : ℝ := 7.43

theorem shorts_cost (shorts_cost : ℝ) : 
  shorts_cost = total_spent - shirt_cost - jacket_cost → shorts_cost = 13.99 := by
  sorry

end shorts_cost_l3168_316892


namespace average_salary_example_l3168_316813

/-- The average salary of 5 people given their individual salaries -/
def average_salary (a b c d e : ℕ) : ℚ :=
  (a + b + c + d + e : ℚ) / 5

/-- Theorem: The average salary of 5 people with salaries 8000, 5000, 14000, 7000, and 9000 is 8200 -/
theorem average_salary_example : average_salary 8000 5000 14000 7000 9000 = 8200 := by
  sorry

#eval average_salary 8000 5000 14000 7000 9000

end average_salary_example_l3168_316813


namespace complex_magnitude_problem_l3168_316875

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem complex_magnitude_problem (z : ℂ) (a : ℝ) 
  (h1 : is_pure_imaginary z) 
  (h2 : (2 + i) * z = 1 + a * i^3) : 
  Complex.abs (a + z) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l3168_316875


namespace factorization_equality_l3168_316855

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end factorization_equality_l3168_316855


namespace gumdrop_purchase_l3168_316816

theorem gumdrop_purchase (total_cents : ℕ) (cost_per_gumdrop : ℕ) (max_gumdrops : ℕ) : 
  total_cents = 224 → cost_per_gumdrop = 8 → max_gumdrops = total_cents / cost_per_gumdrop → max_gumdrops = 28 := by
  sorry

end gumdrop_purchase_l3168_316816


namespace pen_price_calculation_l3168_316895

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 :=
by sorry

end pen_price_calculation_l3168_316895


namespace tetrahedron_circumscribed_sphere_area_l3168_316871

/-- Given a tetrahedron with three mutually perpendicular lateral edges of lengths 1, √2, and √3,
    the surface area of its circumscribed sphere is 6π. -/
theorem tetrahedron_circumscribed_sphere_area (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 2) (h3 : c = Real.sqrt 3) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 6 * Real.pi := by
  sorry

end tetrahedron_circumscribed_sphere_area_l3168_316871


namespace inequality_theorem_l3168_316898

theorem inequality_theorem (a b : ℝ) (h1 : a^3 > b^3) (h2 : a * b < 0) : 1 / a > 1 / b := by
  sorry

end inequality_theorem_l3168_316898


namespace arithmetic_sum_11_l3168_316887

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 11 terms of the arithmetic sequence
    with a₁ = -11 and d = 2 is equal to -11 -/
theorem arithmetic_sum_11 :
  arithmetic_sum (-11) 2 11 = -11 := by
  sorry

end arithmetic_sum_11_l3168_316887


namespace periodic_sum_implies_periodic_components_l3168_316890

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem periodic_sum_implies_periodic_components
  (f g h : ℝ → ℝ) (T : ℝ)
  (h₁ : is_periodic (λ x => f x + g x) T)
  (h₂ : is_periodic (λ x => f x + h x) T)
  (h₃ : is_periodic (λ x => g x + h x) T) :
  is_periodic f T ∧ is_periodic g T ∧ is_periodic h T :=
sorry

end periodic_sum_implies_periodic_components_l3168_316890


namespace opposite_of_negative_hundred_l3168_316811

theorem opposite_of_negative_hundred : -((-100 : ℤ)) = (100 : ℤ) := by
  sorry

end opposite_of_negative_hundred_l3168_316811


namespace min_sum_of_product_1806_l3168_316801

theorem min_sum_of_product_1806 (a b c : ℕ+) : 
  a * b * c = 1806 → 
  (Even a ∨ Even b ∨ Even c) → 
  (∀ x y z : ℕ+, x * y * z = 1806 → (Even x ∨ Even y ∨ Even z) → a + b + c ≤ x + y + z) →
  a + b + c = 112 := by
sorry

end min_sum_of_product_1806_l3168_316801


namespace negation_of_conditional_l3168_316858

theorem negation_of_conditional (x : ℝ) :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) := by sorry

end negation_of_conditional_l3168_316858


namespace max_value_of_expression_max_value_achievable_l3168_316840

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 :=
by sorry

theorem max_value_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) > Real.sqrt 5 / 2 - ε :=
by sorry

end max_value_of_expression_max_value_achievable_l3168_316840


namespace angle_not_sharing_terminal_side_l3168_316848

/-- Two angles share the same terminal side if their difference is a multiple of 360° -/
def ShareTerminalSide (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * k

/-- The main theorem -/
theorem angle_not_sharing_terminal_side :
  let angles : List ℝ := [330, -30, 680, -1110]
  ∀ a ∈ angles, a ≠ 680 → ShareTerminalSide a (-750) ∧
  ¬ ShareTerminalSide 680 (-750) := by
  sorry


end angle_not_sharing_terminal_side_l3168_316848


namespace jose_age_l3168_316817

theorem jose_age (maria_age jose_age : ℕ) : 
  jose_age = maria_age + 12 →
  maria_age + jose_age = 40 →
  jose_age = 26 := by
sorry

end jose_age_l3168_316817


namespace max_value_of_g_l3168_316882

/-- Given a function f(x) = a*cos(x) + b where a and b are constants,
    if the maximum value of f(x) is 1 and the minimum value of f(x) is -7,
    then the maximum value of g(x) = a*cos(x) + b*sin(x) is 5. -/
theorem max_value_of_g (a b : ℝ) :
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) ∧
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) :=
by sorry

end max_value_of_g_l3168_316882


namespace notebook_distribution_l3168_316873

theorem notebook_distribution (class_a class_b notebooks_a notebooks_b : ℕ) 
  (h1 : notebooks_a = class_a / 8)
  (h2 : notebooks_b = 2 * class_a)
  (h3 : 16 = (class_a / 2) / 8)
  (h4 : class_a + class_b = (120 * class_a) / 100) :
  class_a * notebooks_a + class_b * notebooks_b = 2176 := by
  sorry

end notebook_distribution_l3168_316873


namespace line_through_point_l3168_316897

/-- Given a line 2kx - my = 4 passing through the point (3, -2), prove that k = 2/5 and m = 4/5 -/
theorem line_through_point (k m : ℚ) : 
  (2 * k * 3 - m * (-2) = 4) → 
  k = 2/5 ∧ m = 4/5 := by
  sorry

end line_through_point_l3168_316897


namespace missing_sale_is_correct_l3168_316880

/-- Calculates the missing sale amount given sales for 5 out of 6 months and the average sale -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem: The calculated missing sale is correct given the conditions -/
theorem missing_sale_is_correct (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) :
  let sale4 := calculate_missing_sale sale1 sale2 sale3 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#eval calculate_missing_sale 7435 7927 7855 7562 5991 7500

end missing_sale_is_correct_l3168_316880


namespace martha_juice_bottles_l3168_316822

theorem martha_juice_bottles (initial_bottles pantry_bottles fridge_bottles consumed_bottles final_bottles : ℕ) 
  (h1 : initial_bottles = pantry_bottles + fridge_bottles)
  (h2 : pantry_bottles = 4)
  (h3 : fridge_bottles = 4)
  (h4 : consumed_bottles = 3)
  (h5 : final_bottles = 10) : 
  final_bottles - (initial_bottles - consumed_bottles) = 5 := by
  sorry

end martha_juice_bottles_l3168_316822


namespace max_d_value_l3168_316862

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ (552200 + d * 100 + e * 11) % 22 = 0

theorem max_d_value :
  (∃ d e, is_valid_number d e) →
  (∀ d e, is_valid_number d e → d ≤ 6) ∧
  (∃ e, is_valid_number 6 e) :=
by sorry

end max_d_value_l3168_316862


namespace distance_between_stations_distance_is_65km_l3168_316894

/-- The distance between two stations given train travel information -/
theorem distance_between_stations : ℝ :=
let train_p_speed : ℝ := 20
let train_q_speed : ℝ := 25
let train_p_time : ℝ := 2
let train_q_time : ℝ := 1
let distance_p : ℝ := train_p_speed * train_p_time
let distance_q : ℝ := train_q_speed * train_q_time
distance_p + distance_q

/-- Proof that the distance between the stations is 65 km -/
theorem distance_is_65km : distance_between_stations = 65 := by
  sorry

end distance_between_stations_distance_is_65km_l3168_316894


namespace fruit_consumption_l3168_316806

theorem fruit_consumption (total_fruits initial_kept friday_fruits : ℕ) 
  (h_total : total_fruits = 10)
  (h_kept : initial_kept = 2)
  (h_friday : friday_fruits = 3) :
  ∃ (a b o : ℕ),
    a = b ∧ 
    o = 2 * a ∧
    a + b + o = total_fruits - (initial_kept + friday_fruits) ∧
    a = 1 ∧ 
    b = 1 ∧ 
    o = 2 ∧
    a + b + o = 4 := by
  sorry

end fruit_consumption_l3168_316806


namespace count_385_consecutive_sums_l3168_316820

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Predicate for a valid sequence summing to 385 -/
def is_valid_sequence (seq : ConsecutiveSequence) : Prop :=
  sum_consecutive_sequence seq = 385

/-- The main theorem statement -/
theorem count_385_consecutive_sums :
  (∃ (seqs : Finset ConsecutiveSequence), 
    (∀ seq ∈ seqs, is_valid_sequence seq) ∧ 
    (∀ seq, is_valid_sequence seq → seq ∈ seqs) ∧
    seqs.card = 9) := by
  sorry

end count_385_consecutive_sums_l3168_316820


namespace two_circles_congruent_l3168_316863

-- Define the square
def Square := {s : ℝ // s > 0}

-- Define a circle with center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define the configuration of three circles in a square
structure ThreeCirclesInSquare where
  square : Square
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  
  -- Each circle touches two sides of the square
  touches_sides1 : 
    (circle1.center.1 = circle1.radius ∨ circle1.center.1 = square.val - circle1.radius) ∧
    (circle1.center.2 = circle1.radius ∨ circle1.center.2 = square.val - circle1.radius)
  touches_sides2 : 
    (circle2.center.1 = circle2.radius ∨ circle2.center.1 = square.val - circle2.radius) ∧
    (circle2.center.2 = circle2.radius ∨ circle2.center.2 = square.val - circle2.radius)
  touches_sides3 : 
    (circle3.center.1 = circle3.radius ∨ circle3.center.1 = square.val - circle3.radius) ∧
    (circle3.center.2 = circle3.radius ∨ circle3.center.2 = square.val - circle3.radius)

  -- Circles are externally tangent to each other
  externally_tangent12 : (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2
  externally_tangent13 : (circle1.center.1 - circle3.center.1)^2 + (circle1.center.2 - circle3.center.2)^2 = (circle1.radius + circle3.radius)^2
  externally_tangent23 : (circle2.center.1 - circle3.center.1)^2 + (circle2.center.2 - circle3.center.2)^2 = (circle2.radius + circle3.radius)^2

-- Theorem statement
theorem two_circles_congruent (config : ThreeCirclesInSquare) :
  config.circle1.radius = config.circle2.radius ∨ 
  config.circle1.radius = config.circle3.radius ∨ 
  config.circle2.radius = config.circle3.radius :=
sorry

end two_circles_congruent_l3168_316863


namespace number_solution_l3168_316804

theorem number_solution : ∃ x : ℚ, x + (3/5) * x = 240 ∧ x = 150 := by sorry

end number_solution_l3168_316804


namespace orange_juice_serving_size_l3168_316814

/-- Represents the ratio of concentrate to water in the orange juice mixture -/
def concentrateToWaterRatio : ℚ := 1 / 3

/-- The number of cans of concentrate required -/
def concentrateCans : ℕ := 35

/-- The volume of each can of concentrate in ounces -/
def canSize : ℕ := 12

/-- The number of servings to be prepared -/
def numberOfServings : ℕ := 280

/-- The size of each serving in ounces -/
def servingSize : ℚ := 6

theorem orange_juice_serving_size :
  (concentrateCans * canSize * (1 + concentrateToWaterRatio)) / numberOfServings = servingSize :=
sorry

end orange_juice_serving_size_l3168_316814


namespace arithmetic_sequence_common_difference_l3168_316899

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h_d_nonzero : d ≠ 0) 
  (h_sum : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 5 = 13)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ 
    arithmetic_sequence a₁ d 2 = arithmetic_sequence a₁ d 1 * r ∧ 
    arithmetic_sequence a₁ d 5 = arithmetic_sequence a₁ d 2 * r) :
  d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l3168_316899


namespace radical_product_simplification_l3168_316893

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  2 * Real.sqrt (20 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 60 * q * Real.sqrt (30 * q) := by
  sorry

end radical_product_simplification_l3168_316893


namespace smallest_triple_sum_of_squares_l3168_316844

/-- A function that checks if a number can be expressed as the sum of three squares -/
def isSumOfThreeSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a^2 + b^2 + c^2

/-- A function that counts the number of ways a number can be expressed as the sum of three squares -/
def countSumOfThreeSquares (n : ℕ) : ℕ :=
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => 
    let (a, b, c) := triple
    n = a^2 + b^2 + c^2
  ) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1))))).card

/-- Theorem stating that 110 is the smallest positive integer that can be expressed as the sum of three squares in at least three different ways -/
theorem smallest_triple_sum_of_squares : 
  (∀ m : ℕ, m < 110 → countSumOfThreeSquares m < 3) ∧ 
  countSumOfThreeSquares 110 ≥ 3 :=
sorry

end smallest_triple_sum_of_squares_l3168_316844


namespace mika_stickers_l3168_316808

/-- The number of stickers Mika has after all events -/
def final_stickers (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika is left with 28 stickers -/
theorem mika_stickers : final_stickers 45 53 35 19 86 = 28 := by
  sorry

end mika_stickers_l3168_316808


namespace wall_width_l3168_316866

theorem wall_width (w h l : ℝ) (volume : ℝ) : 
  h = 4 * w →
  l = 3 * h →
  volume = w * h * l →
  volume = 10368 →
  w = 6 := by
sorry

end wall_width_l3168_316866


namespace intersection_implies_a_value_l3168_316821

/-- Given sets A and B with the specified elements, if their intersection is {-3},
    then a = -1. -/
theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  (A ∩ B : Set ℝ) = {-3} → a = -1 :=
by
  sorry

end intersection_implies_a_value_l3168_316821


namespace equi_partite_implies_a_equals_two_l3168_316885

/-- A complex number is equi-partite if its real and imaginary parts are equal -/
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

/-- The complex number z in terms of a -/
def z (a : ℝ) : ℂ := (1 + a * Complex.I) - Complex.I

/-- Theorem: If z(a) is an equi-partite complex number, then a = 2 -/
theorem equi_partite_implies_a_equals_two (a : ℝ) :
  is_equi_partite (z a) → a = 2 := by
  sorry


end equi_partite_implies_a_equals_two_l3168_316885


namespace problem_solution_l3168_316877

theorem problem_solution : 
  ∃ x : ℝ, ((15 - 2 + 4) / 2) * x = 77 ∧ x = 77 / 8.5 := by
sorry

end problem_solution_l3168_316877


namespace smallest_square_l3168_316889

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 : ℤ) * a + (16 : ℤ) * b = r^2)
  (h2 : ∃ s : ℕ, (16 : ℤ) * a - (15 : ℤ) * b = s^2) :
  (481 : ℕ)^2 ≤ min ((15 : ℤ) * a + (16 : ℤ) * b) ((16 : ℤ) * a - (15 : ℤ) * b) :=
by sorry

end smallest_square_l3168_316889


namespace complex_to_exponential_form_l3168_316886

theorem complex_to_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 →
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end complex_to_exponential_form_l3168_316886


namespace min_value_theorem_l3168_316843

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 3| - t

-- State the theorem
theorem min_value_theorem (t : ℝ) (a b : ℝ) :
  (∀ x, f t (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 3) →
  (a > 0 ∧ b > 0) →
  (a * b - 2 * a - 8 * b = 2 * t - 2) →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧
    ∀ a' b', a' > 0 → b' > 0 → a' * b' - 2 * a' - 8 * b' = 2 * t - 2 → a₀ + 2 * b₀ ≤ a' + 2 * b') →
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ - 2 * a₀ - 8 * b₀ = 2 * t - 2 ∧ a₀ + 2 * b₀ = 36) :=
by sorry


end min_value_theorem_l3168_316843


namespace test_questions_l3168_316823

theorem test_questions (total_points : ℕ) (four_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : four_point_questions = 10) : 
  ∃ (two_point_questions : ℕ),
    two_point_questions * 2 + four_point_questions * 4 = total_points ∧
    two_point_questions + four_point_questions = 40 := by
  sorry

end test_questions_l3168_316823


namespace inverse_proportion_ratio_l3168_316812

/-- Given that a is inversely proportional to b, prove that b₁/b₂ = 5/4 when a₁/a₂ = 4/5 -/
theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) (hb₁ : b₁ ≠ 0) (hb₂ : b₂ ≠ 0)
    (h_inverse : ∃ k : ℝ, a₁ * b₁ = k ∧ a₂ * b₂ = k) (h_ratio : a₁ / a₂ = 4 / 5) :
  b₁ / b₂ = 5 / 4 := by
  sorry

end inverse_proportion_ratio_l3168_316812


namespace condition_relation_l3168_316800

theorem condition_relation (A B C : Prop) 
  (h1 : B → A)  -- A is a necessary condition for B
  (h2 : C → B)  -- C is a sufficient condition for B
  (h3 : ¬(B → C))  -- C is not a necessary condition for B
  : (C → A) ∧ ¬(A → C) := by
  sorry

end condition_relation_l3168_316800


namespace markers_problem_l3168_316864

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  total_markers = 86 →
  (total_markers - initial_markers) / markers_per_box = 6 :=
by sorry

end markers_problem_l3168_316864


namespace cecilia_always_wins_l3168_316876

theorem cecilia_always_wins (a : ℕ+) : ∃ b : ℕ+, 
  (Nat.gcd a b = 1) ∧ 
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ 
    (p * q * r ∣ a^3 + b^3)) := by
  sorry

end cecilia_always_wins_l3168_316876


namespace todds_profit_l3168_316859

/-- Calculates Todd's remaining money after his snow cone business venture -/
def todds_remaining_money (borrowed : ℕ) (repay : ℕ) (ingredients_cost : ℕ) 
  (num_sold : ℕ) (price_per_cone : ℚ) : ℚ :=
  let total_sales := num_sold * price_per_cone
  let remaining := total_sales - repay
  remaining

/-- Proves that Todd's remaining money is $40 after his snow cone business venture -/
theorem todds_profit : 
  todds_remaining_money 100 110 75 200 (75/100) = 40 := by
  sorry

end todds_profit_l3168_316859


namespace inequality_not_always_true_l3168_316852

theorem inequality_not_always_true 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hc : c ≠ 0) : 
  ∃ c, ¬(a * c > b * c) :=
sorry

end inequality_not_always_true_l3168_316852


namespace sum_of_factors_l3168_316842

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 30 = (x + b) * (x - c)) →
  a + b + c = 12 := by
  sorry

end sum_of_factors_l3168_316842


namespace probability_more_than_seven_is_five_twelfths_l3168_316881

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (totals greater than 7) -/
def favorableOutcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probabilityMoreThanSeven : ℚ := favorableOutcomes / totalOutcomes

theorem probability_more_than_seven_is_five_twelfths :
  probabilityMoreThanSeven = 5 / 12 := by
  sorry

end probability_more_than_seven_is_five_twelfths_l3168_316881


namespace picture_placement_l3168_316837

theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (space_between : ℝ)
  (h1 : wall_width = 25)
  (h2 : picture_width = 2)
  (h3 : space_between = 1)
  (h4 : 2 * picture_width + space_between < wall_width) :
  let distance := (wall_width - (2 * picture_width + space_between)) / 2
  distance = 10 := by
  sorry

end picture_placement_l3168_316837


namespace product_11_4_sum_144_l3168_316829

theorem product_11_4_sum_144 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 11^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 144 :=
by sorry

end product_11_4_sum_144_l3168_316829


namespace butterflies_in_garden_l3168_316841

theorem butterflies_in_garden (initial : ℕ) (flew_away_fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away_fraction = 1/3 → 
  remaining = initial - (initial * flew_away_fraction).num →
  remaining = 6 := by
  sorry

end butterflies_in_garden_l3168_316841


namespace quadratic_inequality_solution_set_l3168_316805

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-(1/2) : ℝ) (1/3) = {x | c * x^2 + b * x + a < 0} := by
sorry

end quadratic_inequality_solution_set_l3168_316805


namespace mark_kate_difference_l3168_316845

/-- Represents the project with three workers -/
structure Project where
  kate_hours : ℕ
  pat_hours : ℕ
  mark_hours : ℕ

/-- Conditions of the project -/
def valid_project (p : Project) : Prop :=
  p.pat_hours = 2 * p.kate_hours ∧
  p.mark_hours = p.kate_hours + 6 ∧
  p.kate_hours + p.pat_hours + p.mark_hours = 198

theorem mark_kate_difference (p : Project) (h : valid_project p) :
  p.mark_hours - p.kate_hours = 6 := by
  sorry

end mark_kate_difference_l3168_316845


namespace geometric_sequence_tan_l3168_316847

/-- Given a geometric sequence {a_n} satisfying certain conditions, 
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tan (a : ℕ → ℝ) : 
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- {a_n} is a geometric sequence
  a 2 * a 3 * a 4 = -a 7^2 →                 -- a_2 * a_3 * a_4 = -a_7^2
  a 2 * a 3 * a 4 = -64 →                    -- a_2 * a_3 * a_4 = -64
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end geometric_sequence_tan_l3168_316847


namespace natural_subset_rational_l3168_316802

theorem natural_subset_rational :
  (∀ x : ℕ, ∃ y : ℚ, (x : ℚ) = y) ∧
  (∃ z : ℚ, ∀ w : ℕ, (w : ℚ) ≠ z) :=
by sorry

end natural_subset_rational_l3168_316802


namespace f_sqrt5_minus1_eq_neg_half_l3168_316870

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

theorem f_sqrt5_minus1_eq_neg_half
  (f : ℝ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : ∀ x > 0, f x * f (f x + 1 / x) = 1) :
  f (Real.sqrt 5 - 1) = -1/2 := by
  sorry

end f_sqrt5_minus1_eq_neg_half_l3168_316870


namespace scientific_notation_correct_l3168_316819

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 1230000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.23
    exponent := 6
    valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end scientific_notation_correct_l3168_316819


namespace tangent_slope_at_origin_l3168_316827

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_slope_at_origin :
  (deriv f) 0 = 1 := by sorry

end tangent_slope_at_origin_l3168_316827


namespace exists_square_composition_function_l3168_316826

theorem exists_square_composition_function : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F ∘ F) n = n^2 := by
  sorry

end exists_square_composition_function_l3168_316826


namespace sqrt_meaningful_iff_geq_neg_one_l3168_316839

theorem sqrt_meaningful_iff_geq_neg_one (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_iff_geq_neg_one_l3168_316839


namespace hyperbola_asymptotes_l3168_316815

/-- Given a hyperbola with equation 9y² - 25x² = 169, 
    its asymptotes are given by the equation y = ± (5/3)x -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 →
  ∃ (k : ℝ), k = 5/3 ∧ (y = k * x ∨ y = -k * x) :=
sorry

end hyperbola_asymptotes_l3168_316815


namespace total_blankets_collected_l3168_316824

/-- Represents the blanket collection problem over three days --/
def blanket_collection (original_members : ℕ) (new_members : ℕ) 
  (blankets_per_original : ℕ) (blankets_per_new : ℕ) 
  (school_blankets : ℕ) (online_blankets : ℕ) : ℕ :=
  let day1 := original_members * blankets_per_original
  let day2_team := original_members * blankets_per_original + new_members * blankets_per_new
  let day2 := day2_team + 3 * day1
  let day3 := school_blankets + online_blankets
  day1 + day2 + day3

/-- The main theorem stating the total number of blankets collected --/
theorem total_blankets_collected : 
  blanket_collection 15 5 2 4 22 30 = 222 := by
  sorry

end total_blankets_collected_l3168_316824


namespace solve_for_x_l3168_316879

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end solve_for_x_l3168_316879


namespace parentheses_make_equations_true_l3168_316865

theorem parentheses_make_equations_true : 
  (5 * (4 + 3) = 35) ∧ (32 / (9 - 5) = 8) := by
  sorry

end parentheses_make_equations_true_l3168_316865


namespace total_fat_ingested_l3168_316878

def fat_content (fish : String) : ℝ :=
  match fish with
  | "herring" => 40
  | "eel" => 20
  | "pike" => 30
  | "salmon" => 35
  | "halibut" => 50
  | _ => 0

def cooking_loss_rate : ℝ := 0.1
def indigestible_rate : ℝ := 0.08

def digestible_fat (fish : String) : ℝ :=
  let initial_fat := fat_content fish
  let after_cooking := initial_fat * (1 - cooking_loss_rate)
  after_cooking * (1 - indigestible_rate)

def fish_counts : List (String × ℕ) := [
  ("herring", 40),
  ("eel", 30),
  ("pike", 25),
  ("salmon", 20),
  ("halibut", 15)
]

theorem total_fat_ingested :
  (fish_counts.map (λ (fish, count) => (digestible_fat fish) * count)).sum = 3643.2 := by
  sorry

end total_fat_ingested_l3168_316878


namespace valid_pairs_l3168_316896

def is_valid_pair (n p : ℕ) : Prop :=
  n > 1 ∧ Nat.Prime p ∧ ((p - 1)^n + 1) % n^(p - 1) = 0

theorem valid_pairs :
  ∀ n p : ℕ, is_valid_pair n p ↔ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end valid_pairs_l3168_316896


namespace base_conversion_1729_l3168_316883

theorem base_conversion_1729 :
  2 * (5 ^ 4) + 3 * (5 ^ 3) + 4 * (5 ^ 2) + 0 * (5 ^ 1) + 4 * (5 ^ 0) = 1729 := by
  sorry

end base_conversion_1729_l3168_316883


namespace find_divisor_l3168_316832

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 217 →
  quotient = 54 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 4 := by
sorry

end find_divisor_l3168_316832


namespace problem_2015_l3168_316872

theorem problem_2015 : (2015^2 + 2015 - 1) / 2015 = 2016 - 1/2015 := by
  sorry

end problem_2015_l3168_316872


namespace equivalent_systems_intersection_l3168_316868

-- Define the type for a linear equation
def LinearEquation := ℝ → ℝ → ℝ

-- Define a system of two linear equations
structure LinearSystem :=
  (eq1 eq2 : LinearEquation)

-- Define the solution set of a linear system
def SolutionSet (sys : LinearSystem) := {p : ℝ × ℝ | sys.eq1 p.1 p.2 = 0 ∧ sys.eq2 p.1 p.2 = 0}

-- Define equivalence of two linear systems
def EquivalentSystems (sys1 sys2 : LinearSystem) :=
  SolutionSet sys1 = SolutionSet sys2

-- Define the intersection points of two lines
def IntersectionPoints (eq1 eq2 : LinearEquation) :=
  {p : ℝ × ℝ | eq1 p.1 p.2 = 0 ∧ eq2 p.1 p.2 = 0}

-- Theorem statement
theorem equivalent_systems_intersection
  (sys1 sys2 : LinearSystem)
  (h : EquivalentSystems sys1 sys2) :
  IntersectionPoints sys1.eq1 sys1.eq2 = IntersectionPoints sys2.eq1 sys2.eq2 := by
  sorry


end equivalent_systems_intersection_l3168_316868


namespace sin_870_degrees_l3168_316825

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by sorry

end sin_870_degrees_l3168_316825


namespace min_value_T_l3168_316860

/-- Given a quadratic inequality that holds for all real x, prove the minimum value of T -/
theorem min_value_T (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  (∀ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' ≥ 0) → a' < b' → 
    (a' + b' + c') / (b' - a') ≥ (a + b + c) / (b - a)) → 
  (a + b + c) / (b - a) = 3 :=
sorry

end min_value_T_l3168_316860


namespace total_cost_theorem_l3168_316861

/-- The total cost of buying thermometers and masks -/
def total_cost (a b : ℝ) : ℝ := 3 * a + b

/-- Theorem: The total cost of buying 3 thermometers at 'a' yuan each
    and 'b' masks at 1 yuan each is equal to (3a + b) yuan -/
theorem total_cost_theorem (a b : ℝ) :
  total_cost a b = 3 * a + b := by sorry

end total_cost_theorem_l3168_316861
