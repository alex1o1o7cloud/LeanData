import Mathlib

namespace sqrt_product_sqrt_l1508_150830

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_product_sqrt_l1508_150830


namespace tuition_calculation_l1508_150810

/-- Given the total cost and the difference between tuition and room and board,
    calculate the tuition fee. -/
theorem tuition_calculation (total_cost room_and_board tuition : ℕ) : 
  total_cost = tuition + room_and_board ∧ 
  tuition = room_and_board + 704 ∧
  total_cost = 2584 →
  tuition = 1644 := by
  sorry

#check tuition_calculation

end tuition_calculation_l1508_150810


namespace perpendicular_sum_l1508_150874

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, then the second component of b is -4. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 3 ∧ b.2 = -2) :
  (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 0 → b.1 = -4 := by
sorry

end perpendicular_sum_l1508_150874


namespace computers_needed_for_expanded_class_l1508_150885

/-- Given an initial number of students, a student-to-computer ratio, and additional students,
    calculate the total number of computers needed to maintain the ratio. -/
def total_computers_needed (initial_students : ℕ) (ratio : ℕ) (additional_students : ℕ) : ℕ :=
  (initial_students / ratio) + (additional_students / ratio)

/-- Theorem: Given 82 initial students, a ratio of 2 students per computer, and 16 additional students,
    the total number of computers needed to maintain the same ratio is 49. -/
theorem computers_needed_for_expanded_class : total_computers_needed 82 2 16 = 49 := by
  sorry

end computers_needed_for_expanded_class_l1508_150885


namespace ellipse_properties_l1508_150855

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and focal distance 2√3 -/
def Ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ a^2 - b^2 = 3

/-- The equation of the ellipse -/
def EllipseEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

/-- Line l₁ with slope k intersecting the ellipse at two points -/
def Line1 (k : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y = k * x ∧ k ≠ 0

/-- Line l₂ with slope k/4 passing through a point on the ellipse -/
def Line2 (k : ℝ) (x₀ y₀ : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ y - y₀ = (k/4) * (x - x₀)

theorem ellipse_properties (a b : ℝ) (h : Ellipse a b) :
  ∃ (k x₀ y₀ x₁ y₁ : ℝ),
    EllipseEquation a b (x₀, y₀) ∧
    EllipseEquation a b (x₁, y₁) ∧
    Line1 k (x₀, y₀) ∧
    Line2 k x₀ y₀ (x₁, y₁) ∧
    (y₁ - y₀) * (x₁ - x₀) = -1/k ∧
    (∀ (x y : ℝ), EllipseEquation a b (x, y) ↔ x^2/4 + y^2 = 1) ∧
    (∃ (M N : ℝ),
      Line2 k x₀ y₀ (M, 0) ∧
      Line2 k x₀ y₀ (0, N) ∧
      ∀ (M' N' : ℝ),
        Line2 k x₀ y₀ (M', 0) ∧
        Line2 k x₀ y₀ (0, N') →
        abs (M * N) / 2 ≤ 9/8) :=
by sorry

end ellipse_properties_l1508_150855


namespace computer_desk_prices_l1508_150880

theorem computer_desk_prices :
  ∃ (x y : ℝ),
    (10 * x + 200 * y = 90000) ∧
    (12 * x + 120 * y = 90000) ∧
    (x = 6000) ∧
    (y = 150) := by
  sorry

end computer_desk_prices_l1508_150880


namespace race_probability_inconsistency_l1508_150814

-- Define the probabilities for each car to win
def prob_X_wins : ℚ := 1/2
def prob_Y_wins : ℚ := 1/4
def prob_Z_wins : ℚ := 1/3

-- Define the total probability of one of them winning
def total_prob : ℚ := 1.0833333333333333

-- Theorem stating the inconsistency of the given probabilities
theorem race_probability_inconsistency :
  prob_X_wins + prob_Y_wins + prob_Z_wins = total_prob ∧
  total_prob > 1 := by sorry

end race_probability_inconsistency_l1508_150814


namespace P_range_l1508_150893

theorem P_range (x : ℝ) (P : ℝ) 
  (h1 : x^2 - 5*x + 6 < 0) 
  (h2 : P = x^2 + 5*x + 6) : 
  20 < P ∧ P < 30 := by
  sorry

end P_range_l1508_150893


namespace student_a_test_questions_l1508_150890

/-- Represents the grading system and test results for Student A -/
structure TestResults where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses

/-- The total number of questions on the test -/
def total_questions (t : TestResults) : ℕ :=
  t.correct_responses + t.incorrect_responses

/-- Theorem stating that the total number of questions on Student A's test is 100 -/
theorem student_a_test_questions :
  ∃ t : TestResults, t.correct_responses = 90 ∧ t.score = 70 ∧ total_questions t = 100 := by
  sorry


end student_a_test_questions_l1508_150890


namespace problem_solution_l1508_150822

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem problem_solution :
  (∀ x : ℝ, a = 1 → (p x a ∧ q x) ↔ x ∈ Set.Ioo 2 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a ∈ Set.Icc 1 2) :=
sorry

end problem_solution_l1508_150822


namespace floor_cube_negative_fraction_l1508_150815

theorem floor_cube_negative_fraction : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end floor_cube_negative_fraction_l1508_150815


namespace possible_value_of_n_l1508_150816

theorem possible_value_of_n : ∃ n : ℕ, 
  3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n ∧ n = 15 := by
  sorry

end possible_value_of_n_l1508_150816


namespace double_discount_price_l1508_150838

-- Define the original price
def original_price : ℝ := 33.78

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the function to apply a discount
def apply_discount (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem statement
theorem double_discount_price :
  apply_discount (apply_discount original_price) = 19.00125 := by
  sorry

end double_discount_price_l1508_150838


namespace mary_next_birthday_age_l1508_150857

theorem mary_next_birthday_age 
  (mary_age sally_age danielle_age : ℝ)
  (h1 : mary_age = 1.25 * sally_age)
  (h2 : sally_age = 0.7 * danielle_age)
  (h3 : mary_age + sally_age + danielle_age = 36) :
  ⌊mary_age⌋ + 1 = 13 :=
by sorry

end mary_next_birthday_age_l1508_150857


namespace largest_multiple_of_8_less_than_60_l1508_150809

theorem largest_multiple_of_8_less_than_60 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 60 → n ≤ 56 :=
by
  sorry

end largest_multiple_of_8_less_than_60_l1508_150809


namespace ratio_equality_l1508_150832

theorem ratio_equality (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_pqr : p^2 + q^2 + r^2 = 49)
  (sum_uvw : u^2 + v^2 + w^2 = 64)
  (dot_product : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_equality_l1508_150832


namespace new_average_age_with_teacher_l1508_150896

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 30 →
  student_avg_age = 14 →
  teacher_age = 45 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) = 15 := by
  sorry

end new_average_age_with_teacher_l1508_150896


namespace symmetric_point_xoy_l1508_150870

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetricPointXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Theorem: The symmetric point of M(m,n,p) with respect to xOy plane is (m,n,-p) -/
theorem symmetric_point_xoy (m n p : ℝ) :
  let M : Point3D := { x := m, y := n, z := p }
  symmetricPointXOY M = { x := m, y := n, z := -p } := by
  sorry

end symmetric_point_xoy_l1508_150870


namespace initial_population_approximation_l1508_150852

/-- The initial population of a town given its final population after a decade of growth. -/
def initial_population (final_population : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  final_population / (1 + growth_rate) ^ years

theorem initial_population_approximation :
  let final_population : ℕ := 297500
  let growth_rate : ℚ := 7 / 100
  let years : ℕ := 10
  ⌊initial_population final_population growth_rate years⌋ = 151195 := by
  sorry

end initial_population_approximation_l1508_150852


namespace lucia_dance_class_cost_l1508_150858

/-- Represents the cost calculation for Lucia's dance classes over a six-week period. -/
def dance_class_cost (hip_hop_cost ballet_cost jazz_cost salsa_cost contemporary_cost : ℚ)
  (hip_hop_freq ballet_freq jazz_freq salsa_freq contemporary_freq : ℚ)
  (extra_salsa_cost : ℚ) : ℚ :=
  hip_hop_cost * hip_hop_freq * 6 +
  ballet_cost * ballet_freq * 6 +
  jazz_cost * jazz_freq * 6 +
  salsa_cost * (6 / salsa_freq) +
  contemporary_cost * (6 / contemporary_freq) +
  extra_salsa_cost

/-- Proves that the total cost of Lucia's dance classes for a six-week period is $465.50. -/
theorem lucia_dance_class_cost :
  dance_class_cost 10.50 12.25 8.75 15 10 3 2 1 2 3 12 = 465.50 := by
  sorry

end lucia_dance_class_cost_l1508_150858


namespace mike_marks_short_l1508_150843

def passing_threshold (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

theorem mike_marks_short (max_marks mike_score : ℕ) 
  (h1 : max_marks = 760) 
  (h2 : mike_score = 212) : 
  passing_threshold max_marks - mike_score = 16 := by
  sorry

end mike_marks_short_l1508_150843


namespace lemon_cupcakes_total_l1508_150876

theorem lemon_cupcakes_total (cupcakes_at_home : ℕ) (boxes_given : ℕ) (cupcakes_per_box : ℕ) : 
  cupcakes_at_home = 2 → boxes_given = 17 → cupcakes_per_box = 3 →
  cupcakes_at_home + boxes_given * cupcakes_per_box = 53 := by
  sorry

end lemon_cupcakes_total_l1508_150876


namespace amusement_park_admission_fee_l1508_150840

theorem amusement_park_admission_fee (child_fee : ℝ) (total_people : ℕ) (total_fee : ℝ) (num_children : ℕ) :
  child_fee = 1.5 →
  total_people = 315 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (adult_fee : ℝ), adult_fee = 4 ∧ 
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fee :=
by
  sorry

end amusement_park_admission_fee_l1508_150840


namespace cork_price_calculation_l1508_150878

/-- The price of a bottle of wine with a cork -/
def bottle_with_cork : ℚ := 2.10

/-- The additional cost of a bottle without a cork compared to the cork price -/
def additional_cost : ℚ := 2.00

/-- The price of the cork -/
def cork_price : ℚ := 0.05

theorem cork_price_calculation :
  cork_price + (cork_price + additional_cost) = bottle_with_cork :=
by sorry

end cork_price_calculation_l1508_150878


namespace smallest_n_divides_l1508_150811

theorem smallest_n_divides (n : ℕ) : n = 90 ↔ 
  (n > 0 ∧ 
   (315^2 - n^2) ∣ (315^3 - n^3) ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬((315^2 - m^2) ∣ (315^3 - m^3))) :=
by sorry

end smallest_n_divides_l1508_150811


namespace area_D_n_formula_l1508_150894

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The region D_n -/
def D_n (n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / (n + 1/2) ≤ p.2 ∧ p.2 ≤ floor (p.1 + 1) - p.1 ∧ p.1 ≥ 0}

/-- The area of D_n -/
noncomputable def area_D_n (n : ℝ) : ℝ := sorry

/-- Theorem: The area of D_n is 1/2 * ((n+3/2)/(n+1/2)) for positive n -/
theorem area_D_n_formula (n : ℝ) (hn : n > 0) :
  area_D_n n = 1/2 * ((n + 3/2) / (n + 1/2)) := by sorry

end area_D_n_formula_l1508_150894


namespace power_equality_l1508_150879

theorem power_equality : 32^5 * 4^3 = 2^31 := by sorry

end power_equality_l1508_150879


namespace race_head_start_l1508_150818

/-- Proves that Cristina gave Nicky a 12-second head start in a 100-meter race -/
theorem race_head_start (race_distance : ℝ) (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) :
  race_distance = 100 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  catch_up_time = 30 →
  (catch_up_time - (nicky_speed * catch_up_time) / cristina_speed) = 12 := by
  sorry

end race_head_start_l1508_150818


namespace simplify_expression_l1508_150825

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a * b^2) - (a * b^2 - b^3) / (a * b^2 - a^3) = (a^3 - a * b^2 + b^4) / (a * b^2) :=
by sorry

end simplify_expression_l1508_150825


namespace cubic_root_ratio_l1508_150867

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) : 
  c / d = -11 / 6 := by
sorry

end cubic_root_ratio_l1508_150867


namespace geometric_sequence_q_eq_one_l1508_150801

/-- A positive geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_q_eq_one
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_prod : a 2 * a 6 = 16)
  (h_sum : a 4 + a 8 = 8) :
  q = 1 := by sorry

end geometric_sequence_q_eq_one_l1508_150801


namespace b_share_is_1000_l1508_150813

/-- Given a partnership with investment ratios A:B:C as 2:2/3:1 and a total profit,
    calculate B's share of the profit. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that given the investment ratios and a total profit of 5500,
    B's share of the profit is 1000. -/
theorem b_share_is_1000 :
  calculate_B_share 5500 = 1000 := by
  sorry

#eval calculate_B_share 5500

end b_share_is_1000_l1508_150813


namespace abs_T_equals_1024_l1508_150812

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end abs_T_equals_1024_l1508_150812


namespace smallest_solution_for_floor_equation_l1508_150841

theorem smallest_solution_for_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 10 ∧
  x = 131 / 11 := by
  sorry

end smallest_solution_for_floor_equation_l1508_150841


namespace log_xy_value_l1508_150829

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x * y) = 8 / 11 := by
  sorry

end log_xy_value_l1508_150829


namespace square_perimeter_increase_l1508_150846

theorem square_perimeter_increase (s : ℝ) : 
  (s + 2) * 4 - s * 4 = 8 := by
sorry

end square_perimeter_increase_l1508_150846


namespace natural_roots_equation_l1508_150866

theorem natural_roots_equation :
  ∃ (x y z t : ℕ),
    17 * (x * y * z * t + x * y + x * t + z * t + 1) - 54 * (y * z * t + y + t) = 0 ∧
    x = 3 ∧ y = 5 ∧ z = 1 ∧ t = 2 :=
by sorry

end natural_roots_equation_l1508_150866


namespace polynomial_factorization_l1508_150824

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (3 * x^2 + 58 * x + 231) * (x + 7) * (x + 11) := by
  sorry

end polynomial_factorization_l1508_150824


namespace carole_wins_iff_n_odd_l1508_150800

/-- The game interval -/
def GameInterval (n : ℕ) := Set.Icc (0 : ℝ) n

/-- Predicate for a valid move -/
def ValidMove (prev : Set ℝ) (x : ℝ) : Prop :=
  ∀ y ∈ prev, |x - y| ≥ 1.5

/-- The game state -/
structure GameState (n : ℕ) where
  chosen : Set ℝ
  current_player : Bool -- true for Carole, false for Leo

/-- The game result -/
inductive GameResult
  | CaroleWins
  | LeoWins

/-- Optimal strategy -/
def OptimalStrategy (n : ℕ) : GameState n → GameResult :=
  sorry

/-- The main theorem -/
theorem carole_wins_iff_n_odd (n : ℕ) (h : n > 10) :
  OptimalStrategy n { chosen := ∅, current_player := true } = GameResult.CaroleWins ↔ Odd n :=
sorry

end carole_wins_iff_n_odd_l1508_150800


namespace weight_difference_is_correct_l1508_150834

/-- The difference in grams between the total weight of oranges and apples -/
def weight_difference : ℝ :=
  let apple_weight_oz : ℝ := 27.5
  let apple_unit_weight_oz : ℝ := 1.5
  let orange_count_dozen : ℝ := 5.5
  let orange_unit_weight_g : ℝ := 45
  let oz_to_g_conversion : ℝ := 28.35

  let apple_weight_g : ℝ := apple_weight_oz * oz_to_g_conversion
  let orange_count : ℝ := orange_count_dozen * 12
  let orange_weight_g : ℝ := orange_count * orange_unit_weight_g

  orange_weight_g - apple_weight_g

theorem weight_difference_is_correct :
  weight_difference = 2190.375 := by
  sorry

end weight_difference_is_correct_l1508_150834


namespace two_fixed_points_l1508_150853

/-- A function satisfying the given property -/
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + x * y + 1

/-- The main theorem -/
theorem two_fixed_points
  (f : ℝ → ℝ)
  (h1 : satisfies_property f)
  (h2 : f (-2) = -2) :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ a : ℤ, a ∈ s ↔ f a = a :=
sorry

end two_fixed_points_l1508_150853


namespace frustum_cross_section_area_l1508_150871

theorem frustum_cross_section_area 
  (S' S Q : ℝ) 
  (n m : ℝ) 
  (h1 : S' > 0) 
  (h2 : S > 0) 
  (h3 : Q > 0) 
  (h4 : n > 0) 
  (h5 : m > 0) :
  Real.sqrt Q = (n * Real.sqrt S + m * Real.sqrt S') / (n + m) := by
sorry

end frustum_cross_section_area_l1508_150871


namespace cyclic_sum_inequality_l1508_150817

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) ≥ 3/2 := by
  sorry

end cyclic_sum_inequality_l1508_150817


namespace original_price_calculation_l1508_150897

/-- Proves that given an article sold for $25 with a gain percent of 150%, the original price of the article was $10. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 ∧ gain_percent = 150 → 
  ∃ (original_price : ℝ), 
    original_price = 10 ∧ 
    selling_price = original_price + (original_price * (gain_percent / 100)) :=
by
  sorry

end original_price_calculation_l1508_150897


namespace cyclist_journey_time_l1508_150882

theorem cyclist_journey_time (a v : ℝ) (h1 : a > 0) (h2 : v > 0) (h3 : a / v = 5) :
  (a / (2 * v)) + (a / (2 * (1.25 * v))) = 4.5 := by
  sorry

end cyclist_journey_time_l1508_150882


namespace golden_ratio_product_ab_pq_minus_n_l1508_150845

/-- The golden ratio is the positive root of x^2 + x - 1 = 0 -/
theorem golden_ratio : ∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2 := by sorry

/-- Given a^2 + ma = 1 and b^2 - 2mb = 4, ab = 2 -/
theorem product_ab (m a b : ℝ) (h1 : a^2 + m*a = 1) (h2 : b^2 - 2*m*b = 4) (h3 : b ≠ -2*a) : a * b = 2 := by sorry

/-- Given p^2 + np - 1 = q and q^2 + nq - 1 = p, pq - n = 0 -/
theorem pq_minus_n (n p q : ℝ) (h1 : p^2 + n*p - 1 = q) (h2 : q^2 + n*q - 1 = p) (h3 : p ≠ q) : p * q - n = 0 := by sorry

end golden_ratio_product_ab_pq_minus_n_l1508_150845


namespace circle_config_exists_l1508_150844

-- Define the type for our circle configuration
def CircleConfig := Fin 8 → Fin 8

-- Define a function to check if two numbers are connected in our configuration
def isConnected (i j : Fin 8) : Prop :=
  (i.val = j.val + 1 ∧ i.val % 2 = 0) ∨
  (j.val = i.val + 1 ∧ j.val % 2 = 0) ∨
  (i.val = j.val + 2 ∧ i.val % 4 = 0) ∨
  (j.val = i.val + 2 ∧ j.val % 4 = 0)

-- Define the property that the configuration satisfies the problem conditions
def validConfig (c : CircleConfig) : Prop :=
  (∀ i : Fin 8, c i ≠ 0) ∧
  (∀ i j : Fin 8, i ≠ j → c i ≠ c j) ∧
  (∀ d : Fin 7, ∃! (i j : Fin 8), isConnected i j ∧ |c i - c j| = d + 1)

-- State the theorem
theorem circle_config_exists : ∃ c : CircleConfig, validConfig c := by
  sorry

end circle_config_exists_l1508_150844


namespace polynomial_subtraction_l1508_150898

/-- Given two polynomials in a and b, prove that their difference is -a^2*b -/
theorem polynomial_subtraction (a b : ℝ) :
  (3 * a^2 * b - 6 * a * b^2) - (2 * a^2 * b - 3 * a * b^2) = -a^2 * b := by
  sorry

end polynomial_subtraction_l1508_150898


namespace closest_fraction_to_japan_medals_l1508_150828

theorem closest_fraction_to_japan_medals :
  let japan_fraction : ℚ := 25 / 120
  let fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
  (1/5 : ℚ) = fractions.argmin (fun x => |x - japan_fraction|) := by
  sorry

end closest_fraction_to_japan_medals_l1508_150828


namespace sine_cosine_shift_l1508_150848

theorem sine_cosine_shift (ω : ℝ) (h_ω : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 8)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x : ℝ, f (x + π / ω) = f x) →
  ∃ k : ℝ, k = 3 * π / 16 ∧ ∀ x : ℝ, g x = f (x + k) :=
by sorry

end sine_cosine_shift_l1508_150848


namespace system_solution_l1508_150805

theorem system_solution (x y : ℝ) : 
  (x^2 + 3*x*y = 12 ∧ x*y = 16 + y^2 - x*y - x^2) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) :=
sorry

end system_solution_l1508_150805


namespace palmer_photos_l1508_150884

/-- The number of photos Palmer has after her trip to Bali -/
def total_photos (initial_photos : ℕ) (first_week : ℕ) (third_fourth_week : ℕ) : ℕ :=
  initial_photos + first_week + 2 * first_week + third_fourth_week

/-- Theorem stating the total number of photos Palmer has after her trip -/
theorem palmer_photos : 
  total_photos 100 50 80 = 330 := by
  sorry

#eval total_photos 100 50 80

end palmer_photos_l1508_150884


namespace double_first_triple_second_row_l1508_150837

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

theorem double_first_triple_second_row (A : Matrix (Fin 2) (Fin 2) ℝ) :
  N • A = !![2 * A 0 0, 2 * A 0 1; 3 * A 1 0, 3 * A 1 1] := by sorry

end double_first_triple_second_row_l1508_150837


namespace houses_traded_l1508_150833

theorem houses_traded (x y z : ℕ) (h : x + y ≥ z) : ∃ t : ℕ, x - t + y = z :=
sorry

end houses_traded_l1508_150833


namespace addition_to_reach_target_l1508_150804

theorem addition_to_reach_target : (1250 / 50) + 7500 = 7525 := by
  sorry

end addition_to_reach_target_l1508_150804


namespace equation_equivalence_l1508_150875

theorem equation_equivalence (a b c : ℝ) (h : a + c = 2 * b) : 
  a^2 + 8 * b * c = (2 * b + c)^2 := by sorry

end equation_equivalence_l1508_150875


namespace vegan_soy_free_fraction_l1508_150803

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_vegan_dishes : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  soy_vegan_dishes = 5 →
  (vegan_dishes - soy_vegan_dishes : ℚ) / total_dishes = 1 / 24 := by
  sorry

end vegan_soy_free_fraction_l1508_150803


namespace hidden_faces_sum_l1508_150806

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

def total_faces : ℕ := 24

theorem hidden_faces_sum (num_dice : ℕ) (h1 : num_dice = 4) :
  num_dice * standard_die_sum - visible_faces.sum = 51 := by
  sorry

end hidden_faces_sum_l1508_150806


namespace inequality_solution_set_l1508_150802

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < 1/2 → S = Set.Ioo a (1 - a)) ∧
  (a = 1/2 → S = ∅) ∧
  (1/2 < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) := by
sorry

end inequality_solution_set_l1508_150802


namespace sum_of_cubes_divisibility_l1508_150888

theorem sum_of_cubes_divisibility (a b c : ℤ) : 
  (3 ∣ (a + b + c)) → (3 ∣ (a^3 + b^3 + c^3)) := by
  sorry

end sum_of_cubes_divisibility_l1508_150888


namespace cement_mixture_weight_l1508_150856

theorem cement_mixture_weight :
  ∀ (total_weight : ℝ),
  (1/5 : ℝ) * total_weight +     -- Weight of sand
  (3/4 : ℝ) * total_weight +     -- Weight of water
  6 = total_weight →             -- Weight of gravel
  total_weight = 120 := by
sorry

end cement_mixture_weight_l1508_150856


namespace circle_tangent_to_x_axis_at_one_zero_l1508_150831

/-- A circle with center (a, a) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The circle is tangent to the x-axis at (1, 0) -/
def isTangentAtOneZero (c : Circle) : Prop :=
  c.r = 1 ∧ c.a = 1

/-- The equation of the circle -/
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.a)^2 = c.r^2

theorem circle_tangent_to_x_axis_at_one_zero :
  ∀ c : Circle, isTangentAtOneZero c →
  ∀ x y : ℝ, circleEquation c x y ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
by sorry

end circle_tangent_to_x_axis_at_one_zero_l1508_150831


namespace simplify_complex_fraction_l1508_150839

theorem simplify_complex_fraction :
  1 / ((2 / (Real.sqrt 2 + 2)) + (3 / (Real.sqrt 3 - 2)) + (4 / (Real.sqrt 5 + 1))) =
  (Real.sqrt 2 + 3 * Real.sqrt 3 - Real.sqrt 5 + 5) / 27 := by sorry

end simplify_complex_fraction_l1508_150839


namespace quadratic_inequality_condition_l1508_150861

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a > 0) ↔ a > 1 := by sorry

end quadratic_inequality_condition_l1508_150861


namespace arithmetic_sequence_problem_l1508_150873

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_2 + a_7 + a_15 = 12,
    prove that a_8 = 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 + a 15 = 12) : a 8 = 4 := by
  sorry

end arithmetic_sequence_problem_l1508_150873


namespace initial_cars_count_l1508_150872

/-- The initial number of cars on the lot -/
def initial_cars : ℕ := sorry

/-- The percentage of initial cars that are silver -/
def initial_silver_percent : ℚ := 1/5

/-- The number of cars in the new shipment -/
def new_shipment : ℕ := 80

/-- The percentage of new cars that are silver -/
def new_silver_percent : ℚ := 1/2

/-- The percentage of total cars that are silver after the new shipment -/
def total_silver_percent : ℚ := 2/5

theorem initial_cars_count : initial_cars = 40 := by
  sorry

end initial_cars_count_l1508_150872


namespace factorization_equality_l1508_150835

theorem factorization_equality (a b : ℝ) : 4*a - a*b^2 = a*(2+b)*(2-b) := by
  sorry

end factorization_equality_l1508_150835


namespace sqrt_three_x_minus_two_lt_x_l1508_150899

theorem sqrt_three_x_minus_two_lt_x (x : ℝ) : 
  Real.sqrt 3 * x - 2 < x ↔ x < Real.sqrt 3 + 1 := by
  sorry

end sqrt_three_x_minus_two_lt_x_l1508_150899


namespace clock_correction_l1508_150847

/-- The daily gain of the clock in minutes -/
def daily_gain : ℚ := 13 / 4

/-- The number of hours between 8 A.M. on April 10 and 3 P.M. on April 19 -/
def total_hours : ℕ := 223

/-- The negative correction in minutes to be subtracted from the clock -/
def m : ℚ := (daily_gain * total_hours) / 24

theorem clock_correction : m = 30 + 13 / 96 := by
  sorry

end clock_correction_l1508_150847


namespace derivative_property_l1508_150823

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem derivative_property (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end derivative_property_l1508_150823


namespace toy_pricing_and_profit_l1508_150886

/-- Represents the order quantity and price for toys -/
structure ToyOrder where
  quantity : ℕ
  price : ℚ

/-- Calculates the factory price based on order quantity -/
def factoryPrice (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else if x < 600 then max (62 - x / 50) 50
  else 50

/-- Calculates the profit for a given order quantity -/
def profit (x : ℕ) : ℚ := (factoryPrice x - 40) * x

theorem toy_pricing_and_profit :
  (∃ x : ℕ, x > 100 ∧ factoryPrice x = 50 → x = 600) ∧
  (∀ x : ℕ, x > 0 → factoryPrice x = 
    if x ≤ 100 then 60
    else if x < 600 then 62 - x / 50
    else 50) ∧
  profit 500 = 6000 := by
  sorry


end toy_pricing_and_profit_l1508_150886


namespace solve_sandwich_problem_l1508_150826

/-- Represents the sandwich eating problem over two days -/
def sandwich_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := (total : ℕ) - first_day.floor - remaining
  first_day.floor - second_day = 2

/-- The theorem representing the sandwich problem -/
theorem solve_sandwich_problem :
  sandwich_problem 12 (1/2) 2 := by
  sorry

end solve_sandwich_problem_l1508_150826


namespace cereal_box_capacity_l1508_150865

theorem cereal_box_capacity (cups_per_serving : ℕ) (total_servings : ℕ) : 
  cups_per_serving = 2 → total_servings = 9 → cups_per_serving * total_servings = 18 := by
  sorry

end cereal_box_capacity_l1508_150865


namespace complex_sum_l1508_150807

theorem complex_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := a + b * i
  z = (1 - i)^2 / (1 + i) →
  a + b = -2 := by
sorry

end complex_sum_l1508_150807


namespace xyz_product_is_27_l1508_150850

theorem xyz_product_is_27 
  (x y z : ℂ) 
  (h1 : x * y + 3 * y = -9)
  (h2 : y * z + 3 * z = -9)
  (h3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by
  sorry

end xyz_product_is_27_l1508_150850


namespace inequality_proof_l1508_150868

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * b * (b + 1) * (c + 1)) + 1 / (b * c * (c + 1) * (a + 1)) + 1 / (c * a * (a + 1) * (b + 1)) ≥ 3 / (1 + a * b * c)^2 := by
  sorry

end inequality_proof_l1508_150868


namespace root_minus_one_implies_k_eq_neg_two_l1508_150854

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) : 
  ((-1 : ℝ)^2 - k * (-1) + 1 = 0) → k = -2 := by
  sorry

end root_minus_one_implies_k_eq_neg_two_l1508_150854


namespace find_x_l1508_150877

theorem find_x : ∃ x : ℚ, (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end find_x_l1508_150877


namespace square_garden_perimeter_l1508_150820

theorem square_garden_perimeter (a p : ℝ) (h1 : a > 0) (h2 : p > 0) (h3 : a = 2 * p + 14.25) : p = 38 := by
  sorry

end square_garden_perimeter_l1508_150820


namespace base7_even_digits_528_l1508_150863

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base7_even_digits_528 :
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end base7_even_digits_528_l1508_150863


namespace internal_curve_convexity_l1508_150821

-- Define a curve as a function from ℝ to ℝ × ℝ
def Curve := ℝ → ℝ × ℝ

-- Define convexity for a curve
def IsConvex (c : Curve) : Prop := sorry

-- Define the r-neighborhood of a curve
def RNeighborhood (c : Curve) (r : ℝ) : Set (ℝ × ℝ) := sorry

-- Define what it means for a curve to bound a set
def Bounds (c : Curve) (s : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem internal_curve_convexity 
  (K : Curve) (r : ℝ) (C : Curve) 
  (h_K_convex : IsConvex K) 
  (h_r_pos : r > 0) 
  (h_C_bounds : Bounds C (RNeighborhood K r)) : 
  IsConvex C := by
  sorry

end internal_curve_convexity_l1508_150821


namespace ellipse_condition_l1508_150862

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  9 * x^2 + y^2 - 18 * x - 2 * y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ (a ≠ b ∨ c ≠ 0 ∨ d ≠ 0) ∧
    ∀ x y : ℝ, curve_equation x y k ↔ a * (x - c)^2 + b * (y - d)^2 = e

/-- The main theorem -/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -10 :=
sorry

end ellipse_condition_l1508_150862


namespace expression_evaluation_l1508_150827

theorem expression_evaluation (a b c : ℚ) 
  (h1 : c = b - 10)
  (h2 : b = a + 2)
  (h3 : a = 4)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 2 ≠ 0)
  (h6 : c + 6 ≠ 0) :
  (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 := by
  sorry

end expression_evaluation_l1508_150827


namespace aj_has_370_stamps_l1508_150869

/-- The number of stamps each person has -/
structure StampCollection where
  aj : ℕ  -- AJ's stamps
  kj : ℕ  -- KJ's stamps
  cj : ℕ  -- CJ's stamps

/-- The conditions of the stamp collection problem -/
def StampProblemConditions (s : StampCollection) : Prop :=
  (s.cj = 2 * s.kj + 5) ∧  -- CJ has 5 more than twice KJ's stamps
  (s.kj = s.aj / 2) ∧      -- KJ has half as many as AJ
  (s.aj + s.kj + s.cj = 930)  -- Total stamps is 930

/-- The theorem stating that AJ has 370 stamps given the conditions -/
theorem aj_has_370_stamps :
  ∀ s : StampCollection, StampProblemConditions s → s.aj = 370 := by
  sorry

end aj_has_370_stamps_l1508_150869


namespace product_sum_difference_l1508_150883

theorem product_sum_difference (a b N : ℤ) : b = 7 → b - a = 2 → a * b = 2 * (a + b) + N → N = 11 := by
  sorry

end product_sum_difference_l1508_150883


namespace total_temp_remaining_days_l1508_150889

/-- Calculates the total temperature of the remaining days in a week given specific conditions. -/
theorem total_temp_remaining_days 
  (avg_temp : ℝ) 
  (days_in_week : ℕ) 
  (temp_first_three : ℝ) 
  (days_first_three : ℕ) 
  (temp_thur_fri : ℝ) 
  (days_thur_fri : ℕ) :
  avg_temp = 60 ∧ 
  days_in_week = 7 ∧ 
  temp_first_three = 40 ∧ 
  days_first_three = 3 ∧ 
  temp_thur_fri = 80 ∧ 
  days_thur_fri = 2 →
  avg_temp * days_in_week - (temp_first_three * days_first_three + temp_thur_fri * days_thur_fri) = 140 :=
by sorry

end total_temp_remaining_days_l1508_150889


namespace heartsuit_three_five_l1508_150849

-- Define the ♥ operation
def heartsuit (x y : ℤ) : ℤ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end heartsuit_three_five_l1508_150849


namespace trig_sum_thirty_degrees_l1508_150819

theorem trig_sum_thirty_degrees :
  let tan30 := Real.sqrt 3 / 3
  let sin30 := 1 / 2
  let cos30 := Real.sqrt 3 / 2
  tan30 + 4 * sin30 + 2 * cos30 = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end trig_sum_thirty_degrees_l1508_150819


namespace average_age_parents_and_children_l1508_150859

theorem average_age_parents_and_children (num_children : ℕ) (num_parents : ℕ) 
  (avg_age_children : ℝ) (avg_age_parents : ℝ) :
  num_children = 40 →
  num_parents = 60 →
  avg_age_children = 12 →
  avg_age_parents = 35 →
  (num_children * avg_age_children + num_parents * avg_age_parents) / (num_children + num_parents) = 25.8 := by
  sorry

end average_age_parents_and_children_l1508_150859


namespace jellybean_probability_l1508_150864

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def jellybeans_picked : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + white_jellybeans) 1) /
  Nat.choose total_jellybeans jellybeans_picked = 14 / 99 := by
  sorry

end jellybean_probability_l1508_150864


namespace arithmetic_geometric_progression_ratio_l1508_150851

/-- Given an arithmetic progression where the k-th, n-th, and p-th terms form three consecutive terms
    of a geometric progression, the common ratio of the geometric progression is (n-p)/(k-n). -/
theorem arithmetic_geometric_progression_ratio
  (a : ℕ → ℝ) -- The arithmetic progression
  (k n p : ℕ) -- Indices of the terms
  (d : ℝ) -- Common difference of the arithmetic progression
  (h1 : ∀ i, a (i + 1) = a i + d) -- Definition of arithmetic progression
  (h2 : ∃ q : ℝ, a n = a k * q ∧ a p = a n * q) -- Geometric progression condition
  : ∃ q : ℝ, q = (n - p) / (k - n) ∧ a n = a k * q ∧ a p = a n * q :=
sorry

end arithmetic_geometric_progression_ratio_l1508_150851


namespace radio_cost_price_l1508_150842

/-- The cost price of a radio given its selling price and loss percentage -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The cost price of a radio sold for 1245 with 17% loss is 1500 -/
theorem radio_cost_price :
  cost_price 1245 17 = 1500 := by
  sorry

end radio_cost_price_l1508_150842


namespace dogs_not_doing_anything_l1508_150881

def total_dogs : ℕ := 264
def running_dogs : ℕ := 40
def playing_dogs : ℕ := 66
def barking_dogs : ℕ := 44
def digging_dogs : ℕ := 26
def agility_dogs : ℕ := 12

theorem dogs_not_doing_anything : 
  total_dogs - (running_dogs + playing_dogs + barking_dogs + digging_dogs + agility_dogs) = 76 := by
  sorry

end dogs_not_doing_anything_l1508_150881


namespace negation_of_existential_quadratic_l1508_150808

theorem negation_of_existential_quadratic (p : Prop) : 
  (p ↔ ∃ x : ℝ, x^2 + 2*x + 2 = 0) → 
  (¬p ↔ ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0) :=
by sorry

end negation_of_existential_quadratic_l1508_150808


namespace jellybean_box_capacity_l1508_150895

theorem jellybean_box_capacity (tim_capacity : ℕ) (scale_factor : ℕ) : 
  tim_capacity = 150 → scale_factor = 3 → 
  (scale_factor ^ 3 : ℕ) * tim_capacity = 4050 := by
  sorry

end jellybean_box_capacity_l1508_150895


namespace geometric_sequence_ratio_l1508_150892

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (seq : GeometricSequence) : ℝ :=
  Classical.choose (seq.is_geometric 1 (by norm_num))

theorem geometric_sequence_ratio 
  (seq : GeometricSequence) 
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  common_ratio seq = 3 := by
sorry

end geometric_sequence_ratio_l1508_150892


namespace store_pricing_strategy_l1508_150860

theorem store_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := 0.7 * list_price
  let marked_price := 1.07 * list_price
  let selling_price := 0.85 * marked_price
  selling_price = 1.3 * purchase_price :=
by sorry

end store_pricing_strategy_l1508_150860


namespace intersection_area_of_bisected_octahedron_l1508_150836

-- Define a regular octahedron
structure RegularOctahedron :=
  (side_length : ℝ)

-- Define the intersection polygon
structure IntersectionPolygon :=
  (octahedron : RegularOctahedron)
  (is_parallel : Bool)
  (is_bisecting : Bool)

-- Define the area of the intersection polygon
def intersection_area (p : IntersectionPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem intersection_area_of_bisected_octahedron 
  (o : RegularOctahedron) 
  (p : IntersectionPolygon) 
  (h1 : o.side_length = 2) 
  (h2 : p.octahedron = o) 
  (h3 : p.is_parallel = true) 
  (h4 : p.is_bisecting = true) : 
  intersection_area p = 9 * Real.sqrt 3 / 8 :=
sorry

end intersection_area_of_bisected_octahedron_l1508_150836


namespace smallest_possible_value_l1508_150891

theorem smallest_possible_value (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 7) →
  (Nat.lcm a b = x * (x + 7)) →
  (a = 56) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, (Nat.gcd 56 c = y + 7) ∧ (Nat.lcm 56 c = y * (y + 7)))) →
  b = 294 := by
  sorry

end smallest_possible_value_l1508_150891


namespace additional_cards_l1508_150887

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) : 
  total_cards = 160 ∧ complete_decks = 3 ∧ cards_per_deck = 52 →
  total_cards - (complete_decks * cards_per_deck) = 4 := by
sorry

end additional_cards_l1508_150887
