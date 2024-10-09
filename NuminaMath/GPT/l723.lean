import Mathlib

namespace find_coordinates_of_B_l723_72303

theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (h1 : ∃ (C1 C2 : ℝ × ℝ), C1.2 = 0 ∧ C2.2 = 0 ∧ (dist C1 A = dist C1 B) ∧ (dist C2 A = dist C2 B) ∧ (A ≠ B))
  (h2 : A = (-3, 2)) :
  B = (-3, -2) :=
sorry

end find_coordinates_of_B_l723_72303


namespace curve_intersects_itself_l723_72390

theorem curve_intersects_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ (t₁^2 - 3, t₁^3 - 6 * t₁ + 4) = (3, 4) ∧ (t₂^2 - 3, t₂^3 - 6 * t₂ + 4) = (3, 4) :=
sorry

end curve_intersects_itself_l723_72390


namespace sport_formulation_water_l723_72330

theorem sport_formulation_water
  (f : ℝ) (c : ℝ) (w : ℝ) 
  (f_s : ℝ) (c_s : ℝ) (w_s : ℝ)
  (standard_ratio : f / c = 1 / 12 ∧ f / w = 1 / 30)
  (sport_ratio_corn_syrup : f_s / c_s = 3 * (f / c))
  (sport_ratio_water : f_s / w_s = (1 / 2) * (f / w))
  (corn_syrup_amount : c_s = 3) :
  w_s = 45 :=
by
  sorry

end sport_formulation_water_l723_72330


namespace part_I_part_II_l723_72323

def sequence_def (x : ℕ → ℝ) (p : ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) = 1 + x n / (p + x n)

theorem part_I (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  p = 2 → ∀ n ∈ (Nat.succ <$> {n | n > 0}), x n < Real.sqrt 2 :=
sorry

theorem part_II (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  (∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) > x n) → ¬ ∃ M ∈ {n | n > 0}, ∀ n > 0, x M ≥ x n :=
sorry

end part_I_part_II_l723_72323


namespace adult_ticket_cost_l723_72318

theorem adult_ticket_cost 
  (child_ticket_cost : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (adults_attended : ℕ)
  (children_tickets : ℕ)
  (adults_ticket_cost : ℕ)
  (h1 : child_ticket_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_cost = 1875)
  (h4 : adults_attended = 175)
  (h5 : children_tickets = total_tickets - adults_attended)
  (h6 : total_cost = adults_attended * adults_ticket_cost + children_tickets * child_ticket_cost) :
  adults_ticket_cost = 9 :=
sorry

end adult_ticket_cost_l723_72318


namespace smallest_integer_solution_l723_72362

theorem smallest_integer_solution (y : ℤ) (h : 7 - 3 * y < 25) : y ≥ -5 :=
by {
  sorry
}

end smallest_integer_solution_l723_72362


namespace point_reflection_l723_72387

-- Definition of point and reflection over x-axis
def P : ℝ × ℝ := (-2, 3)

def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Statement to prove
theorem point_reflection : reflect_x_axis P = (-2, -3) :=
by
  -- Proof goes here
  sorry

end point_reflection_l723_72387


namespace exists_real_polynomial_l723_72317

noncomputable def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, (p.coeff i) < 0

noncomputable def all_positive_coeff (n : ℕ) (p : Polynomial ℝ) : Prop :=
  ∀ i, (Polynomial.derivative^[n] p).coeff i > 0

theorem exists_real_polynomial :
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ (∀ n > 1, all_positive_coeff n p) :=
sorry

end exists_real_polynomial_l723_72317


namespace no_natural_solution_l723_72386

theorem no_natural_solution :
  ¬ (∃ (x y : ℕ), 2 * x + 3 * y = 6) :=
by
sorry

end no_natural_solution_l723_72386


namespace remainder_m_squared_plus_4m_plus_6_l723_72378

theorem remainder_m_squared_plus_4m_plus_6 (m : ℤ) (k : ℤ) (hk : m = 100 * k - 2) :
  (m ^ 2 + 4 * m + 6) % 100 = 2 := 
sorry

end remainder_m_squared_plus_4m_plus_6_l723_72378


namespace speed_of_bus_l723_72341

def distance : ℝ := 500.04
def time : ℝ := 20.0
def conversion_factor : ℝ := 3.6

theorem speed_of_bus :
  (distance / time) * conversion_factor = 90.0072 := 
sorry

end speed_of_bus_l723_72341


namespace line_equation_l723_72388

theorem line_equation (a b : ℝ) (h1 : (1, 2) ∈ line) (h2 : ∃ a b : ℝ, b = 2 * a ∧ line = {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}) :
  line = {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} ∨ line = {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0} :=
sorry

end line_equation_l723_72388


namespace abs_inequality_solution_l723_72345

theorem abs_inequality_solution (x : ℝ) : (|x - 1| < 2) ↔ (x > -1 ∧ x < 3) := 
sorry

end abs_inequality_solution_l723_72345


namespace power_calc_l723_72334

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end power_calc_l723_72334


namespace interest_cannot_be_determined_without_investment_amount_l723_72306

theorem interest_cannot_be_determined_without_investment_amount :
  ∀ (interest_rate : ℚ) (price : ℚ) (invested_amount : Option ℚ),
  interest_rate = 0.16 → price = 128 → invested_amount = none → False :=
by
  sorry

end interest_cannot_be_determined_without_investment_amount_l723_72306


namespace debby_bottles_l723_72365

noncomputable def number_of_bottles_initial : ℕ := 301
noncomputable def number_of_bottles_drank : ℕ := 144
noncomputable def number_of_bottles_left : ℕ := 157

theorem debby_bottles:
  (number_of_bottles_initial - number_of_bottles_drank) = number_of_bottles_left :=
sorry

end debby_bottles_l723_72365


namespace hyperbola_min_focal_asymptote_eq_l723_72359

theorem hyperbola_min_focal_asymptote_eq {x y m : ℝ}
  (h1 : -2 ≤ m)
  (h2 : m < 0)
  (h_eq : x^2 / m^2 - y^2 / (2 * m + 6) = 1)
  (h_min_focal : m = -1) :
  y = 2 * x ∨ y = -2 * x :=
by
  sorry

end hyperbola_min_focal_asymptote_eq_l723_72359


namespace inequality_solution_set_l723_72353

theorem inequality_solution_set : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | x < -1 ∨ x > 5} := 
by
  sorry

end inequality_solution_set_l723_72353


namespace value_range_for_inequality_solution_set_l723_72352

-- Define the condition
def condition (a : ℝ) : Prop := a > 0

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |x - 3| < a

-- State the theorem to be proven
theorem value_range_for_inequality_solution_set (a : ℝ) (h: condition a) : (a > 1) ↔ ∃ x : ℝ, inequality x a := 
sorry

end value_range_for_inequality_solution_set_l723_72352


namespace jack_mopping_time_l723_72329

-- Definitions for the conditions
def bathroom_area : ℝ := 24
def kitchen_area : ℝ := 80
def mopping_rate : ℝ := 8

-- The proof problem: Prove Jack will spend 13 minutes mopping
theorem jack_mopping_time : (bathroom_area + kitchen_area) / mopping_rate = 13 := by
  sorry

end jack_mopping_time_l723_72329


namespace find_product_of_M1_M2_l723_72315

theorem find_product_of_M1_M2 (x M1 M2 : ℝ) 
  (h : (27 * x - 19) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) : 
  M1 * M2 = -2170 := 
sorry

end find_product_of_M1_M2_l723_72315


namespace possible_values_of_q_l723_72379

theorem possible_values_of_q {q : ℕ} (hq : q > 0) :
  (∃ k : ℕ, (5 * q + 35) = k * (3 * q - 7) ∧ k > 0) ↔
  q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 7 ∨ q = 9 ∨ q = 15 ∨ q = 21 ∨ q = 31 :=
by
  sorry

end possible_values_of_q_l723_72379


namespace angle_A_in_triangle_find_b_c_given_a_and_A_l723_72364

theorem angle_A_in_triangle (A B C : ℝ) (a b c : ℝ)
  (h1 : 2 * Real.cos (2 * A) + 4 * Real.cos (B + C) + 3 = 0) :
  A = π / 3 :=
by
  sorry

theorem find_b_c_given_a_and_A (b c : ℝ)
  (A : ℝ)
  (a : ℝ := Real.sqrt 3)
  (h1 : 2 * b * Real.cos A + Real.sqrt (0 - c^2 + 6 * c - 9) = a)
  (h2 : b + c = 3)
  (h3 : A = π / 3) :
  (b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2) :=
by
  sorry

end angle_A_in_triangle_find_b_c_given_a_and_A_l723_72364


namespace quadratic_condition_l723_72373

theorem quadratic_condition (p q : ℝ) (x1 x2 : ℝ) (hx : x1 + x2 = -p) (hq : x1 * x2 = q) :
  p + q = 0 := sorry

end quadratic_condition_l723_72373


namespace problem_divisibility_l723_72356

theorem problem_divisibility (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by
  sorry

end problem_divisibility_l723_72356


namespace Anthony_vs_Jim_l723_72348

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end Anthony_vs_Jim_l723_72348


namespace triangle_area_l723_72332

/-- The area of the triangle enclosed by a line with slope -1/2 passing through (2, -3) and the coordinate axes is 4. -/
theorem triangle_area {l : ℝ → ℝ} (h1 : ∀ x, l x = -1/2 * x + b)
  (h2 : l 2 = -3) : 
  ∃ (A : ℝ) (B : ℝ), 
  ((l 0 = B) ∧ (l A = 0) ∧ (A ≠ 0) ∧ (B ≠ 0)) ∧
  (1/2 * |A| * |B| = 4) := 
sorry

end triangle_area_l723_72332


namespace remainder_of_7_pow_12_mod_100_l723_72338

theorem remainder_of_7_pow_12_mod_100 : (7 ^ 12) % 100 = 1 := 
by sorry

end remainder_of_7_pow_12_mod_100_l723_72338


namespace max_discount_rate_l723_72380

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l723_72380


namespace largest_m_dividing_factorials_l723_72374

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem largest_m_dividing_factorials (m : ℕ) :
  (∀ k : ℕ, k ≤ m → factorial k ∣ (factorial 100 + factorial 99 + factorial 98)) ↔ m = 98 :=
by
  sorry

end largest_m_dividing_factorials_l723_72374


namespace greatest_possible_value_l723_72326

theorem greatest_possible_value :
  ∃ (N P M : ℕ), (M < 10) ∧ (N < 10) ∧ (P < 10) ∧ (M * (111 * M) = N * 1000 + P * 100 + M * 10 + M)
                ∧ (N * 1000 + P * 100 + M * 10 + M = 3996) :=
by
  sorry

end greatest_possible_value_l723_72326


namespace tetrahedron_volume_eq_three_l723_72361

noncomputable def volume_of_tetrahedron : ℝ :=
  let PQ := 3
  let PR := 4
  let PS := 5
  let QR := 5
  let QS := Real.sqrt 34
  let RS := Real.sqrt 41
  have := (PQ = 3) ∧ (PR = 4) ∧ (PS = 5) ∧ (QR = 5) ∧ (QS = Real.sqrt 34) ∧ (RS = Real.sqrt 41)
  3

theorem tetrahedron_volume_eq_three : volume_of_tetrahedron = 3 := 
by { sorry }

end tetrahedron_volume_eq_three_l723_72361


namespace students_answered_both_correctly_l723_72312

theorem students_answered_both_correctly 
(total_students : ℕ) 
(did_not_answer_A_correctly : ℕ) 
(answered_A_correctly_but_not_B : ℕ) 
(h1 : total_students = 50) 
(h2 : did_not_answer_A_correctly = 12) 
(h3 : answered_A_correctly_but_not_B = 30) : 
    (total_students - did_not_answer_A_correctly - answered_A_correctly_but_not_B) = 8 :=
by
    sorry

end students_answered_both_correctly_l723_72312


namespace tan_div_sin_cos_sin_mul_cos_l723_72363

theorem tan_div_sin_cos (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 :=
by
  sorry

theorem sin_mul_cos (α : ℝ) (h : Real.tan α = 7) :
  Real.sin α * Real.cos α = 7 / 50 :=
by
  sorry

end tan_div_sin_cos_sin_mul_cos_l723_72363


namespace gcd_of_factorials_l723_72325

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_of_factorials :
  Nat.gcd (factorial 8) ((factorial 6)^2) = 1440 := by
  sorry

end gcd_of_factorials_l723_72325


namespace coordinates_of_B_l723_72322

theorem coordinates_of_B (m : ℝ) (h : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) :=
by
  -- proof goes here
  sorry

end coordinates_of_B_l723_72322


namespace ellipse_condition_l723_72314

theorem ellipse_condition (k : ℝ) : 
  (k > 1 ↔ 
  (k - 1 > 0 ∧ k + 1 > 0 ∧ k - 1 ≠ k + 1)) :=
by sorry

end ellipse_condition_l723_72314


namespace solve_inequality_l723_72311

theorem solve_inequality (x : ℝ) : 
  (3 * x - 6 > 12 - 2 * x + x^2) ↔ (-1 < x ∧ x < 6) :=
sorry

end solve_inequality_l723_72311


namespace find_number_l723_72393

theorem find_number (x : ℝ) (h : 0.8 * x = (2/5 : ℝ) * 25 + 22) : x = 40 :=
by
  sorry

end find_number_l723_72393


namespace find_three_numbers_l723_72389

theorem find_three_numbers :
  ∃ (x1 x2 x3 k1 k2 k3 : ℕ),
  x1 = 2500 * k1 / (3^k1 - 1) ∧
  x2 = 2500 * k2 / (3^k2 - 1) ∧
  x3 = 2500 * k3 / (3^k3 - 1) ∧
  k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3 ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 :=
by
  sorry

end find_three_numbers_l723_72389


namespace smith_oldest_child_age_l723_72343

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l723_72343


namespace bulls_on_farm_l723_72328

theorem bulls_on_farm (C B : ℕ) (h1 : C / B = 10 / 27) (h2 : C + B = 555) : B = 405 :=
sorry

end bulls_on_farm_l723_72328


namespace relationship_among_a_b_c_l723_72354

noncomputable def f (x : ℝ) : ℝ := sorry  -- The actual function definition is not necessary for this statement.

-- Lean statements for the given conditions
variables {f : ℝ → ℝ}

-- f is even
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- f(x+1) = -f(x)
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 1) = - f x

-- f is monotonically increasing on [-1, 0]
def monotonically_increasing_on (f : ℝ → ℝ) := ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the relationship statement
theorem relationship_among_a_b_c (h1 : even_function f) (h2 : periodic_property f) 
  (h3 : monotonically_increasing_on f) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end relationship_among_a_b_c_l723_72354


namespace arithmetic_sequence_general_term_l723_72331

theorem arithmetic_sequence_general_term (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 7)
  (h_a7 : a 7 = 3) :
  ∀ n, a n = -↑n + 10 :=
by
  sorry

end arithmetic_sequence_general_term_l723_72331


namespace simplify_expression_l723_72381

theorem simplify_expression :
  (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end simplify_expression_l723_72381


namespace areasEqualForHexagonAndOctagon_l723_72395

noncomputable def areaHexagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 6) -- Circumscribed radius
  let a := s / (2 * Real.tan (Real.pi / 6)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

noncomputable def areaOctagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 8) -- Circumscribed radius
  let a := s / (2 * Real.tan (3 * Real.pi / 8)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

theorem areasEqualForHexagonAndOctagon :
  let s := 3
  areaHexagon s = areaOctagon s := sorry

end areasEqualForHexagonAndOctagon_l723_72395


namespace problem_l723_72327

variables (A B C D E : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := E > B ∧ B > D
def condition3 := D > A
def condition4 := C > B

-- Proof goal: Dana (D) and Beth (B) have the same amount of money
theorem problem (h1 : condition1 A C) (h2 : condition2 E B D) (h3 : condition3 D A) (h4 : condition4 C B) : D = B :=
sorry

end problem_l723_72327


namespace geometric_sequence_product_l723_72357

variable {a1 a2 a3 a4 a5 a6 : ℝ}
variable (r : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions defining the terms of a geometric sequence
def is_geometric_sequence (seq : ℕ → ℝ) (a1 r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = seq n * r

-- Given condition: a_3 * a_4 = 5
def given_condition (seq : ℕ → ℝ) := (seq 2 * seq 3 = 5)

-- Proving the required question: a_1 * a_2 * a_5 * a_6 = 5
theorem geometric_sequence_product
  (h_geom : is_geometric_sequence seq a1 r)
  (h_given : given_condition seq) :
  seq 0 * seq 1 * seq 4 * seq 5 = 5 :=
sorry

end geometric_sequence_product_l723_72357


namespace sqrt_sum_inequality_l723_72347

-- Define variables a and b as positive real numbers
variable {a b : ℝ}

-- State the theorem to be proved
theorem sqrt_sum_inequality (ha : 0 < a) (hb : 0 < b) : 
  (a.sqrt + b.sqrt)^8 ≥ 64 * a * b * (a + b)^2 :=
sorry

end sqrt_sum_inequality_l723_72347


namespace largest_square_perimeter_is_28_l723_72351

-- Definitions and assumptions
def rect_length : ℝ := 10
def rect_width : ℝ := 7

-- Define the largest possible square
def largest_square_side := rect_width

-- Define the perimeter of a square
def perimeter_of_square (side : ℝ) : ℝ := 4 * side

-- Proving statement
theorem largest_square_perimeter_is_28 :
  perimeter_of_square largest_square_side = 28 := 
  by 
    -- sorry is used to skip the proof
    sorry

end largest_square_perimeter_is_28_l723_72351


namespace sum_of_integers_l723_72324

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l723_72324


namespace range_of_a_l723_72342

theorem range_of_a
    (a : ℝ)
    (h : ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 4 → x ^ 2 + y ^ 2 = 1) :
    a ∈ (Set.Ioo (-(3 * Real.sqrt 2 / 2)) (-(Real.sqrt 2 / 2)) ∪ Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_a_l723_72342


namespace eight_digit_number_divisible_by_101_l723_72399

def repeat_twice (x : ℕ) : ℕ := 100 * x + x

theorem eight_digit_number_divisible_by_101 (ef gh ij kl : ℕ) 
  (hef : ef < 100) (hgh : gh < 100) (hij : ij < 100) (hkl : kl < 100) :
  (100010001 * repeat_twice ef + 1000010 * repeat_twice gh + 10010 * repeat_twice ij + 10 * repeat_twice kl) % 101 = 0 := sorry

end eight_digit_number_divisible_by_101_l723_72399


namespace train_length_is_correct_l723_72304

noncomputable def lengthOfTrain (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_s

theorem train_length_is_correct : lengthOfTrain 60 15 = 250.05 :=
by
  sorry

end train_length_is_correct_l723_72304


namespace john_mary_game_l723_72309

theorem john_mary_game (n : ℕ) (h : n ≥ 3) :
  ∃ S : ℕ, S = n * (n + 1) :=
by
  sorry

end john_mary_game_l723_72309


namespace exists_four_numbers_product_fourth_power_l723_72335

theorem exists_four_numbers_product_fourth_power :
  ∃ (numbers : Fin 81 → ℕ),
    (∀ i, ∃ a b c : ℕ, numbers i = 2^a * 3^b * 5^c) ∧
    ∃ (i j k l : Fin 81), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ∃ m : ℕ, m^4 = numbers i * numbers j * numbers k * numbers l :=
by
  sorry

end exists_four_numbers_product_fourth_power_l723_72335


namespace calc_expression_l723_72397

theorem calc_expression : (3.242 * 14) / 100 = 0.45388 :=
by
  sorry

end calc_expression_l723_72397


namespace add_to_any_integer_l723_72320

theorem add_to_any_integer (y : ℤ) : (∀ x : ℤ, y + x = x) → y = 0 :=
  by
  sorry

end add_to_any_integer_l723_72320


namespace associates_hired_l723_72350

variable (partners : ℕ) (associates initial_associates hired_associates : ℕ)
variable (initial_ratio : partners / initial_associates = 2 / 63)
variable (final_ratio : partners / (initial_associates + hired_associates) = 1 / 34)
variable (partners_count : partners = 18)

theorem associates_hired : hired_associates = 45 :=
by
  -- Insert solution steps here...
  sorry

end associates_hired_l723_72350


namespace compute_fraction_sum_l723_72392

variable (a b c : ℝ)
open Real

theorem compute_fraction_sum (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -15)
                            (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 6) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 12 := 
sorry

end compute_fraction_sum_l723_72392


namespace inequality_x_alpha_y_beta_l723_72300

theorem inequality_x_alpha_y_beta (x y α β : ℝ) (hx : 0 < x) (hy : 0 < y) 
(hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = 1) : x^α * y^β ≤ α * x + β * y := 
sorry

end inequality_x_alpha_y_beta_l723_72300


namespace range_of_a_l723_72310

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- State the main theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime a x1 = 0 ∧ f_prime a x2 = 0) →
  a < 0 :=
by
  sorry

end range_of_a_l723_72310


namespace range_of_m_l723_72394

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  (A ⊆ B) ↔ (m ≤ (11:ℝ)/3) :=
by
  sorry

end range_of_m_l723_72394


namespace exists_k_divides_poly_l723_72305

theorem exists_k_divides_poly (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : a 2 = 1) 
  (h₂ : ∀ k : ℕ, a (k + 2) = a (k + 1) + a k) :
  ∀ (m : ℕ), m > 0 → ∃ k : ℕ, m ∣ (a k ^ 4 - a k - 2) :=
by
  sorry

end exists_k_divides_poly_l723_72305


namespace unique_solution_to_equation_l723_72302

theorem unique_solution_to_equation (a : ℝ) (h : ∀ x : ℝ, a * x^2 + Real.sin x ^ 2 = a^2 - a) : a = 1 :=
sorry

end unique_solution_to_equation_l723_72302


namespace boat_breadth_is_two_l723_72376

noncomputable def breadth_of_boat (L h m g ρ : ℝ) : ℝ :=
  let W := m * g
  let V := W / (ρ * g)
  V / (L * h)

theorem boat_breadth_is_two :
  breadth_of_boat 7 0.01 140 9.81 1000 = 2 := 
by
  unfold breadth_of_boat
  simp
  sorry

end boat_breadth_is_two_l723_72376


namespace license_plates_count_l723_72371

-- Definitions from conditions
def num_digits : ℕ := 4
def num_digits_choices : ℕ := 10
def num_letters : ℕ := 3
def num_letters_choices : ℕ := 26

-- Define the blocks and their possible arrangements
def digits_permutations : ℕ := num_digits_choices^num_digits
def letters_permutations : ℕ := num_letters_choices^num_letters
def block_positions : ℕ := 5

-- We need to show that total possible license plates is 878,800,000.
def total_plates : ℕ := digits_permutations * letters_permutations * block_positions

-- The theorem statement
theorem license_plates_count :
  total_plates = 878800000 := by
  sorry

end license_plates_count_l723_72371


namespace expected_waiting_time_correct_l723_72344

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l723_72344


namespace number_of_people_is_8_l723_72385

noncomputable def find_number_of_people (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ) (n : ℕ) :=
  avg_increase = weight_diff / n ∧ old_weight = 70 ∧ new_weight = 90 ∧ weight_diff = new_weight - old_weight → n = 8

theorem number_of_people_is_8 :
  ∃ n : ℕ, find_number_of_people 2.5 70 90 20 n :=
by
  use 8
  sorry

end number_of_people_is_8_l723_72385


namespace fraction_identity_l723_72358

def at_op (a b : ℤ) : ℤ := a * b - 3 * b ^ 2
def hash_op (a b : ℤ) : ℤ := a + 2 * b - 2 * a * b ^ 2

theorem fraction_identity : (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end fraction_identity_l723_72358


namespace infinite_not_expressible_as_sum_of_three_squares_l723_72383

theorem infinite_not_expressible_as_sum_of_three_squares :
  ∃ (n : ℕ), ∃ (infinitely_many_n : ℕ → Prop), (∀ m:ℕ, (infinitely_many_n m ↔ m ≡ 7 [MOD 8])) ∧ ∀ a b c : ℕ, n ≠ a^2 + b^2 + c^2 := 
by
  sorry

end infinite_not_expressible_as_sum_of_three_squares_l723_72383


namespace paco_salty_cookies_left_l723_72313

-- Define the initial number of salty cookies Paco had
def initial_salty_cookies : ℕ := 26

-- Define the number of salty cookies Paco ate
def eaten_salty_cookies : ℕ := 9

-- The theorem statement that Paco had 17 salty cookies left
theorem paco_salty_cookies_left : initial_salty_cookies - eaten_salty_cookies = 17 := 
 by
  -- Here we skip the proof by adding sorry
  sorry

end paco_salty_cookies_left_l723_72313


namespace curve_transformation_l723_72301

theorem curve_transformation (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →
  (x' = 4 * x) →
  (y' = 2 * y) →
  (x'^2 / 16 + y'^2 / 4 = 1) :=
by
  sorry

end curve_transformation_l723_72301


namespace blue_paint_needed_l723_72367

theorem blue_paint_needed (F B : ℝ) :
  (6/9 * F = 4/5 * (F * 1/3 + B) → B = 1/2 * F) :=
sorry

end blue_paint_needed_l723_72367


namespace sum_of_squares_l723_72339

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 23) (h2 : a * b + b * c + a * c = 131) :
  a^2 + b^2 + c^2 = 267 :=
by
  sorry

end sum_of_squares_l723_72339


namespace sports_club_problem_l723_72340

theorem sports_club_problem (total_members : ℕ) (members_playing_badminton : ℕ) 
  (members_playing_tennis : ℕ) (members_not_playing_either : ℕ) 
  (h_total_members : total_members = 100) (h_badminton : members_playing_badminton = 60) 
  (h_tennis : members_playing_tennis = 70) (h_neither : members_not_playing_either = 10) : 
  (members_playing_badminton + members_playing_tennis - 
   (total_members - members_not_playing_either) = 40) :=
by {
  sorry
}

end sports_club_problem_l723_72340


namespace find_least_d_l723_72319

theorem find_least_d :
  ∃ d : ℕ, (d % 7 = 1) ∧ (d % 5 = 2) ∧ (d % 3 = 2) ∧ d = 92 :=
by 
  sorry

end find_least_d_l723_72319


namespace perfect_square_trinomial_m_l723_72346

theorem perfect_square_trinomial_m (m : ℤ) (x : ℤ) : (∃ a : ℤ, x^2 - mx + 16 = (x - a)^2) ↔ (m = 8 ∨ m = -8) :=
by sorry

end perfect_square_trinomial_m_l723_72346


namespace arithmetic_sequence_common_difference_l723_72398

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 3 = 7) (h2 : a 7 = -5)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = -3 :=
sorry

end arithmetic_sequence_common_difference_l723_72398


namespace cost_of_5kg_l723_72370

def cost_of_seeds (x : ℕ) : ℕ :=
  if x ≤ 2 then 5 * x else 4 * x + 2

theorem cost_of_5kg : cost_of_seeds 5 = 22 := by
  sorry

end cost_of_5kg_l723_72370


namespace compute_volume_of_cube_l723_72308

-- Define the conditions and required properties
variable (s V : ℝ)

-- Given condition: the surface area of the cube is 384 sq cm
def surface_area (s : ℝ) : Prop := 6 * s^2 = 384

-- Define the volume of the cube
def volume (s : ℝ) (V : ℝ) : Prop := V = s^3

-- Theorem statement to prove the volume is correctly computed
theorem compute_volume_of_cube (h₁ : surface_area s) : volume s 512 :=
  sorry

end compute_volume_of_cube_l723_72308


namespace add_water_to_solution_l723_72377

noncomputable def current_solution_volume : ℝ := 300
noncomputable def desired_water_percentage : ℝ := 0.70
noncomputable def current_water_volume : ℝ := 0.60 * current_solution_volume
noncomputable def current_acid_volume : ℝ := 0.40 * current_solution_volume

theorem add_water_to_solution (x : ℝ) : 
  (current_water_volume + x) / (current_solution_volume + x) = desired_water_percentage ↔ x = 100 :=
by
  sorry

end add_water_to_solution_l723_72377


namespace negative_solution_condition_l723_72375

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l723_72375


namespace fewest_printers_l723_72337

theorem fewest_printers (x y : ℕ) (h1 : 375 * x = 150 * y) : x + y = 7 :=
  sorry

end fewest_printers_l723_72337


namespace train_speed_in_kmph_l723_72360

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end train_speed_in_kmph_l723_72360


namespace tangerines_left_l723_72349

def total_tangerines : ℕ := 27
def tangerines_eaten : ℕ := 18

theorem tangerines_left : total_tangerines - tangerines_eaten = 9 := by
  sorry

end tangerines_left_l723_72349


namespace Grace_minus_Lee_l723_72333

-- Definitions for the conditions
def Grace_calculation : ℤ := 12 - (3 * 4 - 2)
def Lee_calculation : ℤ := (12 - 3) * 4 - 2

-- Statement of the problem to prove
theorem Grace_minus_Lee : Grace_calculation - Lee_calculation = -32 := by
  sorry

end Grace_minus_Lee_l723_72333


namespace slope_point_on_line_l723_72321

theorem slope_point_on_line (b : ℝ) (h1 : ∃ x, x + b = 30) (h2 : (b / (30 - b)) = 4) : b = 24 :=
  sorry

end slope_point_on_line_l723_72321


namespace initial_blocks_l723_72316

-- Definitions of the given conditions
def blocks_eaten : ℕ := 29
def blocks_remaining : ℕ := 26

-- The statement we need to prove
theorem initial_blocks : blocks_eaten + blocks_remaining = 55 :=
by
  -- Proof is not required as per instructions
  sorry

end initial_blocks_l723_72316


namespace lattice_points_distance_5_l723_72307

def is_lattice_point (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 = 25

theorem lattice_points_distance_5 : 
  ∃ S : Finset (ℤ × ℤ × ℤ), 
    (∀ p ∈ S, is_lattice_point p.1 p.2.1 p.2.2) ∧
    S.card = 78 :=
by
  sorry

end lattice_points_distance_5_l723_72307


namespace number_of_ordered_pairs_l723_72336

theorem number_of_ordered_pairs (p q : ℂ) (h1 : p^4 * q^3 = 1) (h2 : p^8 * q = 1) : (∃ n : ℕ, n = 40) :=
sorry

end number_of_ordered_pairs_l723_72336


namespace polynomial_g_l723_72396

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l723_72396


namespace ratio_apples_pie_to_total_is_one_to_two_l723_72391

variable (x : ℕ) -- number of apples Paul put aside for pie
variable (total_apples : ℕ := 62) 
variable (fridge_apples : ℕ := 25)
variable (muffin_apples : ℕ := 6)

def apples_pie_ratio (x total_apples : ℕ) : ℕ := x / gcd x total_apples

theorem ratio_apples_pie_to_total_is_one_to_two :
  x + fridge_apples + muffin_apples = total_apples -> apples_pie_ratio x total_apples = 1 / 2 :=
by
  sorry

end ratio_apples_pie_to_total_is_one_to_two_l723_72391


namespace triangle_inequality_equality_iff_equilateral_l723_72369

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem equality_iff_equilateral (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_inequality_equality_iff_equilateral_l723_72369


namespace product_not_end_in_1999_l723_72366

theorem product_not_end_in_1999 (a b c d e : ℕ) (h : a + b + c + d + e = 200) : 
  ¬(a * b * c * d * e % 10000 = 1999) := 
by
  sorry

end product_not_end_in_1999_l723_72366


namespace min_value_of_a_l723_72372

theorem min_value_of_a 
  (a b x1 x2 : ℕ) 
  (h1 : a = b - 2005) 
  (h2 : (x1 + x2) = a) 
  (h3 : (x1 * x2) = b) 
  (h4 : x1 > 0 ∧ x2 > 0) : 
  a ≥ 95 :=
sorry

end min_value_of_a_l723_72372


namespace total_treats_is_237_l723_72368

def num_children : ℕ := 3
def hours_out : ℕ := 4
def houses_visited (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 4
  | 2 => 6
  | 3 => 5
  | 4 => 7
  | _ => 0

def treats_per_kid_per_house (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 3
  | 3 => 3
  | 2 => 4
  | 4 => 4
  | _ => 0

def total_treats : ℕ :=
  (houses_visited 1 * treats_per_kid_per_house 1 * num_children) + 
  (houses_visited 2 * treats_per_kid_per_house 2 * num_children) +
  (houses_visited 3 * treats_per_kid_per_house 3 * num_children) +
  (houses_visited 4 * treats_per_kid_per_house 4 * num_children)

theorem total_treats_is_237 : total_treats = 237 :=
by
  -- Placeholder for the proof
  sorry

end total_treats_is_237_l723_72368


namespace paintable_wall_area_correct_l723_72355

noncomputable def paintable_wall_area : Nat :=
  let length := 15
  let width := 11
  let height := 9
  let closet_width := 3
  let closet_length := 4
  let unused_area := 70
  let room_wall_area :=
    2 * (length * height) +
    2 * (width * height)
  let closet_wall_area := 
    2 * (closet_width * height)
  let paintable_area_per_bedroom := 
    room_wall_area - (unused_area + closet_wall_area)
  4 * paintable_area_per_bedroom

theorem paintable_wall_area_correct : paintable_wall_area = 1376 := by
  sorry

end paintable_wall_area_correct_l723_72355


namespace num_sides_of_length4_eq_4_l723_72384

-- Definitions of the variables and conditions
def total_sides : ℕ := 6
def total_perimeter : ℕ := 30
def side_length1 : ℕ := 7
def side_length2 : ℕ := 4

-- The conditions imposed by the problem
def is_hexagon (x y : ℕ) : Prop := x + y = total_sides
def perimeter_condition (x y : ℕ) : Prop := side_length1 * x + side_length2 * y = total_perimeter

-- The proof problem: Prove that the number of sides of length 4 is 4
theorem num_sides_of_length4_eq_4 (x y : ℕ) 
    (h1 : is_hexagon x y) 
    (h2 : perimeter_condition x y) : y = 4 :=
sorry

end num_sides_of_length4_eq_4_l723_72384


namespace solve_for_m_l723_72382

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → - (1/2) * x^2 + 2 * x > m * x) → m = 1 :=
by
  -- Skip the proof by using sorry
  sorry

end solve_for_m_l723_72382
