import Mathlib

namespace NUMINAMATH_GPT_polygon_sides_l1295_129559

theorem polygon_sides (n : ℕ) 
  (h1 : sum_interior_angles = 180 * (n - 2))
  (h2 : sum_exterior_angles = 360)
  (h3 : sum_interior_angles = 3 * sum_exterior_angles) : 
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1295_129559


namespace NUMINAMATH_GPT_d_is_greatest_l1295_129577

variable (p : ℝ)

def a := p - 1
def b := p + 2
def c := p - 3
def d := p + 4

theorem d_is_greatest : d > b ∧ d > a ∧ d > c := 
by sorry

end NUMINAMATH_GPT_d_is_greatest_l1295_129577


namespace NUMINAMATH_GPT_find_dividend_l1295_129574

theorem find_dividend (x D : ℕ) (q r : ℕ) (h_q : q = 4) (h_r : r = 3)
  (h_div : D = x * q + r) (h_sum : D + x + q + r = 100) : D = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_dividend_l1295_129574


namespace NUMINAMATH_GPT_inequality_solution_l1295_129548

theorem inequality_solution (a b : ℝ)
  (h₁ : ∀ x, - (1 : ℝ) / 2 < x ∧ x < (1 : ℝ) / 3 → ax^2 + bx + (2 : ℝ) > 0)
  (h₂ : - (1 : ℝ) / 2 = -(b / a))
  (h₃ : (- (1 : ℝ) / 6) = 2 / a) :
  a - b = -10 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1295_129548


namespace NUMINAMATH_GPT_wolf_nobel_laureates_l1295_129552

/-- 31 scientists that attended a certain workshop were Wolf Prize laureates,
and some of them were also Nobel Prize laureates. Of the scientists who attended
that workshop and had not received the Wolf Prize, the number of scientists who had
received the Nobel Prize was 3 more than the number of scientists who had not received
the Nobel Prize. In total, 50 scientists attended that workshop, and 25 of them were
Nobel Prize laureates. Prove that the number of Wolf Prize laureates who were also
Nobel Prize laureates is 3. -/
theorem wolf_nobel_laureates (W N total W' N' W_N : ℕ)  
  (hW : W = 31) (hN : N = 25) (htotal : total = 50) 
  (hW' : W' = total - W) (hN' : N' = total - N) 
  (hcondition : N' - W' = 3) :
  W_N = N - W' :=
by
  sorry

end NUMINAMATH_GPT_wolf_nobel_laureates_l1295_129552


namespace NUMINAMATH_GPT_original_amount_of_cooking_oil_l1295_129585

theorem original_amount_of_cooking_oil (X : ℝ) (H : (2 / 5 * X + 300) + (1 / 2 * (X - (2 / 5 * X + 300)) - 200) + 800 = X) : X = 2500 :=
by simp at H; linarith

end NUMINAMATH_GPT_original_amount_of_cooking_oil_l1295_129585


namespace NUMINAMATH_GPT_percy_swimming_hours_l1295_129591

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end NUMINAMATH_GPT_percy_swimming_hours_l1295_129591


namespace NUMINAMATH_GPT_hexagon_AF_length_l1295_129538

theorem hexagon_AF_length (BC CD DE EF : ℝ) (angleB angleC angleD angleE : ℝ) (angleF : ℝ) 
  (hBC : BC = 2) (hCD : CD = 2) (hDE : DE = 2) (hEF : EF = 2)
  (hangleB : angleB = 135) (hangleC : angleC = 135) (hangleD : angleD = 135) (hangleE : angleE = 135)
  (hangleF : angleF = 90) :
  ∃ (a b : ℝ), (AF = a + 2 * Real.sqrt b) ∧ (a + b = 6) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_AF_length_l1295_129538


namespace NUMINAMATH_GPT_an_squared_diff_consec_cubes_l1295_129570

theorem an_squared_diff_consec_cubes (a b : ℕ → ℤ) (n : ℕ) :
  a 1 = 1 → b 1 = 0 →
  (∀ n ≥ 1, a (n + 1) = 7 * (a n) + 12 * (b n) + 6) →
  (∀ n ≥ 1, b (n + 1) = 4 * (a n) + 7 * (b n) + 3) →
  a n ^ 2 = (b n + 1) ^ 3 - (b n) ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_an_squared_diff_consec_cubes_l1295_129570


namespace NUMINAMATH_GPT_find_b3_b17_l1295_129535

variable {a : ℕ → ℤ} -- Arithmetic sequence
variable {b : ℕ → ℤ} -- Geometric sequence

axiom arith_seq {a : ℕ → ℤ} (d : ℤ) : ∀ (n : ℕ), a (n + 1) = a n + d
axiom geom_seq {b : ℕ → ℤ} (r : ℤ) : ∀ (n : ℕ), b (n + 1) = b n * r

theorem find_b3_b17 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) 
  (h_geom : ∃ r, ∀ n, b (n + 1) = b n * r)
  (h_cond1 : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10) :
  b 3 * b 17 = 36 := 
sorry

end NUMINAMATH_GPT_find_b3_b17_l1295_129535


namespace NUMINAMATH_GPT_triangle_ab_value_l1295_129530

theorem triangle_ab_value (a b c : ℝ) (A B C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ab_value_l1295_129530


namespace NUMINAMATH_GPT_exists_k_for_any_n_l1295_129517

theorem exists_k_for_any_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, 2 * k^2 + 2001 * k + 3 ≡ 0 [MOD 2^n] :=
sorry

end NUMINAMATH_GPT_exists_k_for_any_n_l1295_129517


namespace NUMINAMATH_GPT_analytical_expression_f_min_value_f_range_of_k_l1295_129504

noncomputable def max_real (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
  max_real (|x + 1|) (|x - 2|)

noncomputable def g (x k : ℝ) : ℝ :=
  x^2 - k * f x

-- Problem 1: Proving the analytical expression of f(x)
theorem analytical_expression_f (x : ℝ) :
  f x = if x < 0.5 then 2 - x else x + 1 :=
sorry

-- Problem 2: Proving the minimum value of f(x)
theorem min_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 3 / 2 :=
sorry

-- Problem 3: Proving the range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (g x k) ≤ (g (x - 1) k)) → k ≤ 2 :=
sorry

end NUMINAMATH_GPT_analytical_expression_f_min_value_f_range_of_k_l1295_129504


namespace NUMINAMATH_GPT_simple_interest_rate_l1295_129575

theorem simple_interest_rate (P : ℝ) (r : ℝ) (T : ℝ) (SI : ℝ)
  (h1 : SI = P / 5)
  (h2 : T = 10)
  (h3 : SI = (P * r * T) / 100) :
  r = 2 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1295_129575


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1295_129550

variable (a b c : ℝ)

theorem quadratic_inequality_solution_set (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 → (-1 / 3 < x ∧ x < 2)) :
  ∀ x : ℝ, cx^2 + bx + a < 0 → (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1295_129550


namespace NUMINAMATH_GPT_find_k_l1295_129599

variables (a b : ℝ × ℝ)
variables (k : ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, -1)

def k_a_plus_b (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def a_minus_2b : ℝ × ℝ := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_product (k_a_plus_b k) a_minus_2b = 0 ↔ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1295_129599


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l1295_129541

noncomputable def common_ratio (a₁ S₃ : ℕ) : ℕ := by sorry

theorem arithmetic_sequence_n_value:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
  (∀ n, a n > 0) →
  a 1 = 3 →
  S 3 = 21 →
  (∃ q, q > 0 ∧ common_ratio 1 q = q ∧ a 5 = 48) →
  n = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l1295_129541


namespace NUMINAMATH_GPT_problem_1_problem_2_l1295_129532

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem problem_1 (m : ℝ) (h_mono : ∀ x y, m ≤ x → x ≤ y → y ≤ m + 1 → f y ≤ f x) : m ≤ 1 :=
  sorry

theorem problem_2 (a b : ℝ) (h_min : a < b) 
  (h_min_val : ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x)
  (h_max_val : ∀ x, a ≤ x ∧ x ≤ b → f x ≤ f b) 
  (h_fa_eq_a : f a = a) (h_fb_eq_b : f b = b) : a = 2 ∧ b = 3 :=
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1295_129532


namespace NUMINAMATH_GPT_total_percentage_increase_l1295_129598

noncomputable def initialSalary : ℝ := 60
noncomputable def firstRaisePercent : ℝ := 10
noncomputable def secondRaisePercent : ℝ := 15
noncomputable def promotionRaisePercent : ℝ := 20

theorem total_percentage_increase :
  let finalSalary := initialSalary * (1 + firstRaisePercent / 100) * (1 + secondRaisePercent / 100) * (1 + promotionRaisePercent / 100)
  let increase := finalSalary - initialSalary
  let percentageIncrease := (increase / initialSalary) * 100
  percentageIncrease = 51.8 := by
  sorry

end NUMINAMATH_GPT_total_percentage_increase_l1295_129598


namespace NUMINAMATH_GPT_sum_of_numbers_odd_probability_l1295_129503

namespace ProbabilityProblem

/-- 
  Given a biased die where the probability of rolling an even number is 
  twice the probability of rolling an odd number, and rolling the die three times,
  the probability that the sum of the numbers rolled is odd is 13/27.
-/
theorem sum_of_numbers_odd_probability :
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let prob_all_odd := (p_odd) ^ 3
  let prob_one_odd_two_even := 3 * (p_odd) * (p_even) ^ 2
  prob_all_odd + prob_one_odd_two_even = 13 / 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_odd_probability_l1295_129503


namespace NUMINAMATH_GPT_admission_price_for_adults_l1295_129543

def total_people := 610
def num_adults := 350
def child_price := 1
def total_receipts := 960

theorem admission_price_for_adults (A : ℝ) (h1 : 350 * A + 260 = 960) : A = 2 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_admission_price_for_adults_l1295_129543


namespace NUMINAMATH_GPT_scrabble_score_l1295_129563

-- Definitions derived from conditions
def value_first_and_third : ℕ := 1
def value_middle : ℕ := 8
def multiplier : ℕ := 3

-- Prove the total points earned by Jeremy
theorem scrabble_score : (value_first_and_third * 2 + value_middle) * multiplier = 30 :=
by
  sorry

end NUMINAMATH_GPT_scrabble_score_l1295_129563


namespace NUMINAMATH_GPT_grassy_pathway_area_correct_l1295_129557

-- Define the dimensions of the plot and the pathway width
def length_plot : ℝ := 15
def width_plot : ℝ := 10
def width_pathway : ℝ := 2

-- Define the required areas
def total_area : ℝ := (length_plot + 2 * width_pathway) * (width_plot + 2 * width_pathway)
def plot_area : ℝ := length_plot * width_plot
def grassy_pathway_area : ℝ := total_area - plot_area

-- Prove that the area of the grassy pathway is 116 m²
theorem grassy_pathway_area_correct : grassy_pathway_area = 116 := by
  sorry

end NUMINAMATH_GPT_grassy_pathway_area_correct_l1295_129557


namespace NUMINAMATH_GPT_Camp_Cedar_number_of_counselors_l1295_129507

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_Camp_Cedar_number_of_counselors_l1295_129507


namespace NUMINAMATH_GPT_min_value_of_w_l1295_129565

noncomputable def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w : ∃ x y : ℝ, ∀ (a b : ℝ), w x y ≤ w a b ∧ w x y = 19 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_w_l1295_129565


namespace NUMINAMATH_GPT_smallest_possible_z_l1295_129546

theorem smallest_possible_z :
  ∃ (z : ℕ), (z = 6) ∧ 
  ∃ (u w x y : ℕ), u < w ∧ w < x ∧ x < y ∧ y < z ∧ 
  u.succ = w ∧ w.succ = x ∧ x.succ = y ∧ y.succ = z ∧ 
  u^3 + w^3 + x^3 + y^3 = z^3 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_smallest_possible_z_l1295_129546


namespace NUMINAMATH_GPT_find_a_l1295_129505

-- Given conditions
def div_by_3 (a : ℤ) : Prop :=
  (5 * a + 1) % 3 = 0 ∨ (3 * a + 2) % 3 = 0

def div_by_5 (a : ℤ) : Prop :=
  (5 * a + 1) % 5 = 0 ∨ (3 * a + 2) % 5 = 0

-- Proving the question 
theorem find_a (a : ℤ) : div_by_3 a ∧ div_by_5 a → a % 15 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1295_129505


namespace NUMINAMATH_GPT_geometric_sequence_tenth_term_l1295_129589

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (4 / 3 : ℚ)
  a * r ^ 9 = (1048576 / 19683 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_tenth_term_l1295_129589


namespace NUMINAMATH_GPT_simplify_polynomial_sum_l1295_129558

/- Define the given polynomials -/
def polynomial1 (x : ℝ) : ℝ := (5 * x^10 + 8 * x^9 + 3 * x^8)
def polynomial2 (x : ℝ) : ℝ := (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9)
def resultant_polynomial (x : ℝ) : ℝ := (2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9)

theorem simplify_polynomial_sum (x : ℝ) :
  polynomial1 x + polynomial2 x = resultant_polynomial x :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_sum_l1295_129558


namespace NUMINAMATH_GPT_gcd_poly_l1295_129519

-- Defining the conditions
def is_odd_multiple_of_17 (b : ℤ) : Prop := ∃ k : ℤ, b = 17 * (2 * k + 1)

theorem gcd_poly (b : ℤ) (h : is_odd_multiple_of_17 b) : 
  Int.gcd (12 * b^3 + 7 * b^2 + 49 * b + 106) 
          (3 * b + 7) = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_poly_l1295_129519


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l1295_129567

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l1295_129567


namespace NUMINAMATH_GPT_julia_tulip_count_l1295_129516

def tulip_count (tulips daisies : ℕ) : Prop :=
  3 * daisies = 7 * tulips

theorem julia_tulip_count : 
  ∃ t, tulip_count t 65 ∧ t = 28 := 
by
  sorry

end NUMINAMATH_GPT_julia_tulip_count_l1295_129516


namespace NUMINAMATH_GPT_average_velocity_of_particle_l1295_129506

theorem average_velocity_of_particle (t : ℝ) (s : ℝ → ℝ) (h_s : ∀ t, s t = t^2 + 1) :
  (s 2 - s 1) / (2 - 1) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_velocity_of_particle_l1295_129506


namespace NUMINAMATH_GPT_number_of_members_l1295_129551

theorem number_of_members (n : ℕ) (h1 : ∀ m : ℕ, m = n → m * m = 1936) : n = 44 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_members_l1295_129551


namespace NUMINAMATH_GPT_find_length_of_room_l1295_129523

noncomputable def cost_of_paving : ℝ := 21375
noncomputable def rate_per_sq_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem find_length_of_room :
  ∃ l : ℝ, l = (cost_of_paving / rate_per_sq_meter) / width_of_room ∧ l = 5 := by
  sorry

end NUMINAMATH_GPT_find_length_of_room_l1295_129523


namespace NUMINAMATH_GPT_seashells_given_to_Joan_l1295_129582

def S_original : ℕ := 35
def S_now : ℕ := 17

theorem seashells_given_to_Joan :
  (S_original - S_now) = 18 := by
  sorry

end NUMINAMATH_GPT_seashells_given_to_Joan_l1295_129582


namespace NUMINAMATH_GPT_additional_vegetables_can_be_planted_l1295_129531

-- Defines the garden's initial conditions.
def tomatoes_kinds := 3
def tomatoes_each := 5
def cucumbers_kinds := 5
def cucumbers_each := 4
def potatoes := 30
def rows := 10
def spaces_per_row := 15

-- The proof statement.
theorem additional_vegetables_can_be_planted (total_tomatoes : ℕ := tomatoes_kinds * tomatoes_each)
                                              (total_cucumbers : ℕ := cucumbers_kinds * cucumbers_each)
                                              (total_potatoes : ℕ := potatoes)
                                              (total_spaces : ℕ := rows * spaces_per_row) :
  total_spaces - (total_tomatoes + total_cucumbers + total_potatoes) = 85 := 
by 
  sorry

end NUMINAMATH_GPT_additional_vegetables_can_be_planted_l1295_129531


namespace NUMINAMATH_GPT_internet_plan_comparison_l1295_129522

theorem internet_plan_comparison (d : ℕ) :
    3000 + 200 * d > 5000 → d > 10 :=
by
  intro h
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_internet_plan_comparison_l1295_129522


namespace NUMINAMATH_GPT_find_total_original_cost_l1295_129500

noncomputable def original_total_cost (x y z : ℝ) : ℝ :=
x + y + z

theorem find_total_original_cost (x y z : ℝ)
  (h1 : x * 1.30 = 351)
  (h2 : y * 1.25 = 275)
  (h3 : z * 1.20 = 96) :
  original_total_cost x y z = 570 :=
sorry

end NUMINAMATH_GPT_find_total_original_cost_l1295_129500


namespace NUMINAMATH_GPT_ch4_contains_most_atoms_l1295_129555

def molecule_atoms (molecule : String) : Nat :=
  match molecule with
  | "O₂"   => 2
  | "NH₃"  => 4
  | "CO"   => 2
  | "CH₄"  => 5
  | _      => 0

theorem ch4_contains_most_atoms :
  ∀ (a b c d : Nat), 
  a = molecule_atoms "O₂" →
  b = molecule_atoms "NH₃" →
  c = molecule_atoms "CO" →
  d = molecule_atoms "CH₄" →
  d > a ∧ d > b ∧ d > c :=
by
  intros
  sorry

end NUMINAMATH_GPT_ch4_contains_most_atoms_l1295_129555


namespace NUMINAMATH_GPT_alternating_colors_probability_l1295_129545

theorem alternating_colors_probability :
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_outcomes : ℕ := 2
  let total_outcomes : ℕ := Nat.choose total_balls white_balls
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := 
by
  let total_balls := 10
  let white_balls := 5
  let black_balls := 5
  let successful_outcomes := 2
  let total_outcomes := Nat.choose total_balls white_balls
  have h_total_outcomes : total_outcomes = 252 := sorry
  have h_probability : (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := sorry
  exact h_probability

end NUMINAMATH_GPT_alternating_colors_probability_l1295_129545


namespace NUMINAMATH_GPT_roots_of_polynomial_l1295_129529

theorem roots_of_polynomial :
  {r : ℝ | (10 * r^4 - 55 * r^3 + 96 * r^2 - 55 * r + 10 = 0)} = {2, 1, 1 / 2} :=
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l1295_129529


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l1295_129511

-- Define the conditions and the question
theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 30) (h2 : a^2 / (1 - r^2) = 180) : a = 10 :=
by sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l1295_129511


namespace NUMINAMATH_GPT_exists_n_for_sin_l1295_129533

theorem exists_n_for_sin (x : ℝ) (h : Real.sin x ≠ 0) :
  ∃ n : ℕ, |Real.sin (n * x)| ≥ Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_exists_n_for_sin_l1295_129533


namespace NUMINAMATH_GPT_tomorrowIsUncertain_l1295_129515

-- Definitions as conditions
def isCertainEvent (e : Prop) : Prop := e = true
def isImpossibleEvent (e : Prop) : Prop := e = false
def isInevitableEvent (e : Prop) : Prop := e = true
def isUncertainEvent (e : Prop) : Prop := e ≠ true ∧ e ≠ false

-- Event: Tomorrow will be sunny
def tomorrowWillBeSunny : Prop := sorry -- Placeholder for the actual weather prediction model

-- Problem statement: Prove that "Tomorrow will be sunny" is an uncertain event
theorem tomorrowIsUncertain : isUncertainEvent tomorrowWillBeSunny := sorry

end NUMINAMATH_GPT_tomorrowIsUncertain_l1295_129515


namespace NUMINAMATH_GPT_second_shift_fraction_of_total_l1295_129594

theorem second_shift_fraction_of_total (W E : ℕ) (h1 : ∀ (W : ℕ), E = (3 * W / 4))
  : let W₁ := W
    let E₁ := E
    let widgets_first_shift := W₁ * E₁
    let widgets_per_second_shift_employee := (2 * W₁) / 3
    let second_shift_employees := (4 * E₁) / 3
    let widgets_second_shift := (2 * W₁ / 3) * (4 * E₁ / 3)
    let total_widgets := widgets_first_shift + widgets_second_shift
    let fraction_second_shift := widgets_second_shift / total_widgets
    fraction_second_shift = 8 / 17 :=
sorry

end NUMINAMATH_GPT_second_shift_fraction_of_total_l1295_129594


namespace NUMINAMATH_GPT_problem_equivalent_to_l1295_129573

theorem problem_equivalent_to (x : ℝ)
  (A : x^2 = 5*x - 6 ↔ x = 2 ∨ x = 3)
  (B : x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3)
  (C : x = x + 1 ↔ false)
  (D : x^2 - 5*x + 7 = 1 ↔ x = 2 ∨ x = 3)
  (E : x^2 - 1 = 5*x - 7 ↔ x = 2 ∨ x = 3) :
  ¬ (x = x + 1) :=
by sorry

end NUMINAMATH_GPT_problem_equivalent_to_l1295_129573


namespace NUMINAMATH_GPT_license_plate_increase_l1295_129537

theorem license_plate_increase :
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  new_plates / old_plates = (900 / 17576) * 100 :=
by
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  have h : new_plates / old_plates = (900 / 17576) * 100 := sorry
  exact h

end NUMINAMATH_GPT_license_plate_increase_l1295_129537


namespace NUMINAMATH_GPT_greatest_integer_x_l1295_129539

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_x_l1295_129539


namespace NUMINAMATH_GPT_find_number_l1295_129593

noncomputable def calc1 : Float := 0.47 * 1442
noncomputable def calc2 : Float := 0.36 * 1412
noncomputable def diff : Float := calc1 - calc2

theorem find_number :
  ∃ (n : Float), (diff + n = 6) :=
sorry

end NUMINAMATH_GPT_find_number_l1295_129593


namespace NUMINAMATH_GPT_overall_ratio_men_women_l1295_129536

variables (m_w_diff players_total beginners_m beginners_w intermediate_m intermediate_w advanced_m advanced_w : ℕ)

def total_men : ℕ := beginners_m + intermediate_m + advanced_m
def total_women : ℕ := beginners_w + intermediate_w + advanced_w

theorem overall_ratio_men_women 
  (h1 : beginners_m = 2) 
  (h2 : beginners_w = 4)
  (h3 : intermediate_m = 3) 
  (h4 : intermediate_w = 5) 
  (h5 : advanced_m = 1) 
  (h6 : advanced_w = 3) 
  (h7 : m_w_diff = 4)
  (h8 : total_men = 6)
  (h9 : total_women = 12)
  (h10 : players_total = 18) :
  total_men / total_women = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_overall_ratio_men_women_l1295_129536


namespace NUMINAMATH_GPT_integer_values_between_fractions_l1295_129578

theorem integer_values_between_fractions :
  let a := 4 / (Real.sqrt 3 + Real.sqrt 2)
  let b := 4 / (Real.sqrt 5 - Real.sqrt 3)
  ((⌊b⌋ - ⌈a⌉) + 1) = 6 :=
by sorry

end NUMINAMATH_GPT_integer_values_between_fractions_l1295_129578


namespace NUMINAMATH_GPT_determine_ABC_l1295_129554

-- Define values in the new base system
def base_representation (A B C : ℕ) : ℕ :=
  A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + C * (A+1)^4 + B * (A+1)^3 + B * (A+1)^2 + B * (A+1) + C

-- The conditions given by the problem
def condition (A B C : ℕ) : Prop :=
  ((A+1)^8 - 2*(A+1)^4 + 1) = base_representation A B C

-- The theorem to be proved
theorem determine_ABC : ∃ (A B C : ℕ), A = 2 ∧ B = 0 ∧ C = 1 ∧ condition A B C :=
by
  existsi 2
  existsi 0
  existsi 1
  unfold condition base_representation
  sorry

end NUMINAMATH_GPT_determine_ABC_l1295_129554


namespace NUMINAMATH_GPT_simplify_fraction_l1295_129581

theorem simplify_fraction : 
  (5 / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108)) = (5 * Real.sqrt 3 / 54) :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1295_129581


namespace NUMINAMATH_GPT_integral_evaluation_l1295_129525

noncomputable def integral_result : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) - x)

theorem integral_evaluation :
  integral_result = (Real.pi - 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_integral_evaluation_l1295_129525


namespace NUMINAMATH_GPT_total_copies_to_save_40_each_l1295_129521

-- Definitions for the conditions.
def cost_per_copy : ℝ := 0.02
def discount_rate : ℝ := 0.25
def min_copies_for_discount : ℕ := 100
def savings_required : ℝ := 0.40
def steve_copies : ℕ := 80
def dinley_copies : ℕ := 80

-- Lean 4 statement to prove the total number of copies 
-- to save $0.40 each.
theorem total_copies_to_save_40_each : 
  (steve_copies + dinley_copies) + 
  (savings_required / (cost_per_copy * discount_rate)) * 2 = 320 :=
by 
  sorry

end NUMINAMATH_GPT_total_copies_to_save_40_each_l1295_129521


namespace NUMINAMATH_GPT_calculate_3_diamond_4_l1295_129583

-- Define the operations
def op (a b : ℝ) : ℝ := a^2 + 2 * a * b
def diamond (a b : ℝ) : ℝ := 4 * a + 6 * b - op a b

-- State the theorem
theorem calculate_3_diamond_4 : diamond 3 4 = 3 := by
  sorry

end NUMINAMATH_GPT_calculate_3_diamond_4_l1295_129583


namespace NUMINAMATH_GPT_find_value_of_a_l1295_129502

theorem find_value_of_a (a : ℝ) (h : 0.005 * a = 65) : a = 130 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1295_129502


namespace NUMINAMATH_GPT_smallest_odd_factors_gt_100_l1295_129520

theorem smallest_odd_factors_gt_100 : ∃ n : ℕ, n > 100 ∧ (∀ d : ℕ, d ∣ n → (∃ m : ℕ, n = m * m)) ∧ (∀ m : ℕ, m > 100 ∧ (∀ d : ℕ, d ∣ m → (∃ k : ℕ, m = k * k)) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_odd_factors_gt_100_l1295_129520


namespace NUMINAMATH_GPT_non_congruent_triangles_perimeter_18_l1295_129579

theorem non_congruent_triangles_perimeter_18 :
  ∃ (triangles : Finset (Finset ℕ)), triangles.card = 11 ∧
  (∀ t ∈ triangles, t.card = 3 ∧ (∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 18 ∧ a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end NUMINAMATH_GPT_non_congruent_triangles_perimeter_18_l1295_129579


namespace NUMINAMATH_GPT_find_positive_product_l1295_129556

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end NUMINAMATH_GPT_find_positive_product_l1295_129556


namespace NUMINAMATH_GPT_inletRate_is_3_l1295_129587

def volumeTank (v_cubic_feet : ℕ) : ℕ :=
  1728 * v_cubic_feet

def outletRate1 : ℕ := 9 -- rate of first outlet in cubic inches/min
def outletRate2 : ℕ := 6 -- rate of second outlet in cubic inches/min
def tankVolume : ℕ := volumeTank 30 -- tank volume in cubic inches
def minutesToEmpty : ℕ := 4320 -- time to empty the tank in minutes

def effectiveRate (inletRate : ℕ) : ℕ :=
  outletRate1 + outletRate2 - inletRate

theorem inletRate_is_3 : (15 - 3) * minutesToEmpty = tankVolume :=
  by simp [outletRate1, outletRate2, tankVolume, minutesToEmpty]; sorry

end NUMINAMATH_GPT_inletRate_is_3_l1295_129587


namespace NUMINAMATH_GPT_cow_value_increase_l1295_129527

theorem cow_value_increase :
  let starting_weight : ℝ := 732
  let increase_factor : ℝ := 1.35
  let price_per_pound : ℝ := 2.75
  let new_weight := starting_weight * increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  let increase_in_value := value_at_new_weight - value_at_starting_weight
  increase_in_value = 704.55 :=
by
  sorry

end NUMINAMATH_GPT_cow_value_increase_l1295_129527


namespace NUMINAMATH_GPT_wrongly_noted_mark_is_90_l1295_129544

-- Define the given conditions
def avg_marks (n : ℕ) (avg : ℚ) : ℚ := n * avg

def wrong_avg_marks : ℚ := avg_marks 10 100
def correct_avg_marks : ℚ := avg_marks 10 92

-- Equate the difference caused by the wrong mark
theorem wrongly_noted_mark_is_90 (x : ℚ) (h₁ : wrong_avg_marks = 1000) (h₂ : correct_avg_marks = 920) (h : x - 10 = 1000 - 920) : x = 90 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_wrongly_noted_mark_is_90_l1295_129544


namespace NUMINAMATH_GPT_incorrect_option_C_l1295_129596

-- Definitions of increasing and decreasing functions
def increasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂
def decreasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- The incorrectness of option C
theorem incorrect_option_C (f g : ℝ → ℝ) 
  (h₁ : increasing f) 
  (h₂ : decreasing g) : ¬ increasing (fun x => f x + g x) := 
sorry

end NUMINAMATH_GPT_incorrect_option_C_l1295_129596


namespace NUMINAMATH_GPT_probability_top_king_of_hearts_l1295_129524

def deck_size : ℕ := 52

def king_of_hearts_count : ℕ := 1

def probability_king_of_hearts_top_card (n : ℕ) (k : ℕ) : ℚ :=
  if n ≠ 0 then k / n else 0

theorem probability_top_king_of_hearts : 
  probability_king_of_hearts_top_card deck_size king_of_hearts_count = 1 / 52 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_top_king_of_hearts_l1295_129524


namespace NUMINAMATH_GPT_angle_bounds_find_configurations_l1295_129510

/-- Given four points A, B, C, D on a plane, where α1 and α2 are the two smallest angles,
    and β1 and β2 are the two largest angles formed by these points, we aim to prove:
    1. 0 ≤ α2 ≤ 45 degrees,
    2. 72 degrees ≤ β2 ≤ 180 degrees,
    and to find configurations that achieve α2 = 45 degrees and β2 = 72 degrees. -/
theorem angle_bounds {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ) 
  (h_angles : α1 ≤ α2 ∧ α2 ≤ β2 ∧ β2 ≤ β1 ∧ 
              0 ≤ α2 ∧ α2 ≤ 45 ∧ 
              72 ≤ β2 ∧ β2 ≤ 180) : 
  (0 ≤ α2 ∧ α2 ≤ 45 ∧ 72 ≤ β2 ∧ β2 ≤ 180) := 
by sorry

/-- Find configurations where α2 = 45 degrees and β2 = 72 degrees. -/
theorem find_configurations {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ)
  (h_angles : α1 ≤ α2 ∧ α2 = 45 ∧ β2 = 72 ∧ β2 ≤ β1) :
  (α2 = 45 ∧ β2 = 72) := 
by sorry

end NUMINAMATH_GPT_angle_bounds_find_configurations_l1295_129510


namespace NUMINAMATH_GPT_fraction_of_students_older_than_4_years_l1295_129562

-- Definitions based on conditions
def total_students := 50
def students_younger_than_3 := 20
def students_not_between_3_and_4 := 25
def students_older_than_4 := students_not_between_3_and_4 - students_younger_than_3
def fraction_older_than_4 := students_older_than_4 / total_students

-- Theorem to prove the desired fraction
theorem fraction_of_students_older_than_4_years : fraction_older_than_4 = 1/10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_older_than_4_years_l1295_129562


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1295_129580

theorem minimum_value_of_expression
  (a b c : ℝ)
  (h : 2 * a + 2 * b + c = 8) :
  ∃ x, (x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2) ∧ x ≥ (49 / 9) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1295_129580


namespace NUMINAMATH_GPT_percentage_increase_B_more_than_C_l1295_129572

noncomputable def percentage_increase :=
  let C_m := 14000
  let A_annual := 470400
  let A_m := A_annual / 12
  let B_m := (2 / 5) * A_m
  ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_more_than_C : percentage_increase = 12 :=
  sorry

end NUMINAMATH_GPT_percentage_increase_B_more_than_C_l1295_129572


namespace NUMINAMATH_GPT_seashells_initial_count_l1295_129595

theorem seashells_initial_count (S : ℝ) (h : S + 4.0 = 10) : S = 6.0 :=
by
  sorry

end NUMINAMATH_GPT_seashells_initial_count_l1295_129595


namespace NUMINAMATH_GPT_find_multiple_l1295_129564

theorem find_multiple (n : ℕ) (h₁ : n = 5) (m : ℕ) (h₂ : 7 * n - 15 > m * n) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1295_129564


namespace NUMINAMATH_GPT_total_cost_correct_l1295_129526

-- Define the costs for each day
def day1_rate : ℝ := 150
def day1_miles_cost : ℝ := 0.50 * 620
def gps_service_cost : ℝ := 10
def day1_total_cost : ℝ := day1_rate + day1_miles_cost + gps_service_cost

def day2_rate : ℝ := 100
def day2_miles_cost : ℝ := 0.40 * 744
def day2_total_cost : ℝ := day2_rate + day2_miles_cost + gps_service_cost

def day3_rate : ℝ := 75
def day3_miles_cost : ℝ := 0.30 * 510
def day3_total_cost : ℝ := day3_rate + day3_miles_cost + gps_service_cost

-- Define the total cost
def total_cost : ℝ := day1_total_cost + day2_total_cost + day3_total_cost

-- Prove that the total cost is equal to the calculated value
theorem total_cost_correct : total_cost = 1115.60 :=
by
  -- This is where the proof would go, but we leave it out for now
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1295_129526


namespace NUMINAMATH_GPT_proof_problem_l1295_129553

open Real

noncomputable def problem_condition1 (A B : ℝ) : Prop :=
  (sin A - sin B) * (sin A + sin B) = sin (π/3 - B) * sin (π/3 + B)

noncomputable def problem_condition2 (b c : ℝ) (a : ℝ) (dot_product : ℝ) : Prop :=
  b * c * cos (π / 3) = dot_product ∧ a = 2 * sqrt 7

noncomputable def problem_condition3 (a b c : ℝ) : Prop := 
  a^2 = (b + c)^2 - 3 * b * c

noncomputable def problem_condition4 (b c : ℝ) : Prop := 
  b < c

theorem proof_problem (A B : ℝ) (a b c dot_product : ℝ)
  (h1 : problem_condition1 A B)
  (h2 : problem_condition2 b c a dot_product)
  (h3 : problem_condition3 a b c)
  (h4 : problem_condition4 b c) :
  (A = π / 3) ∧ (b = 4 ∧ c = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1295_129553


namespace NUMINAMATH_GPT_total_paint_remaining_l1295_129534

-- Definitions based on the conditions
def paint_per_statue : ℚ := 1 / 16
def statues_to_paint : ℕ := 14

-- Theorem statement to prove the answer
theorem total_paint_remaining : (statues_to_paint : ℚ) * paint_per_statue = 7 / 8 := 
by sorry

end NUMINAMATH_GPT_total_paint_remaining_l1295_129534


namespace NUMINAMATH_GPT_total_tickets_needed_l1295_129540

-- Define the conditions
def rollercoaster_rides (n : Nat) := 3
def catapult_rides (n : Nat) := 2
def ferris_wheel_rides (n : Nat) := 1
def rollercoaster_cost (n : Nat) := 4
def catapult_cost (n : Nat) := 4
def ferris_wheel_cost (n : Nat) := 1

-- Prove the total number of tickets needed
theorem total_tickets_needed : 
  rollercoaster_rides 0 * rollercoaster_cost 0 +
  catapult_rides 0 * catapult_cost 0 +
  ferris_wheel_rides 0 * ferris_wheel_cost 0 = 21 :=
by 
  sorry

end NUMINAMATH_GPT_total_tickets_needed_l1295_129540


namespace NUMINAMATH_GPT_minimum_value_N_div_a4_possible_values_a4_l1295_129518

noncomputable def lcm_10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) : ℕ := 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a1 a2) a3) a4) a5) a6) a7) a8) a9) a10

theorem minimum_value_N_div_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10) : 
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 := sorry

theorem possible_values_a4 {a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ} 
  (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6 ∧ a6 < a7 ∧ a7 < a8 ∧ a8 < a9 ∧ a9 < a10)
  (z: 1 ≤ a4 ∧ a4 ≤ 1300) :
  (lcm_10 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10) / a4 = 630 → a4 = 360 ∨ a4 = 720 ∨ a4 = 1080 := sorry

end NUMINAMATH_GPT_minimum_value_N_div_a4_possible_values_a4_l1295_129518


namespace NUMINAMATH_GPT_range_of_m_for_subset_l1295_129590

open Set

variable (m : ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | (2 * m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_of_m_for_subset (m : ℝ) : B m ⊆ A ↔ m ∈ Icc (-(1 / 2) : ℝ) (2 : ℝ) ∨ m > (2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_subset_l1295_129590


namespace NUMINAMATH_GPT_johns_pace_l1295_129592

variable {J : ℝ} -- John's pace during his final push

theorem johns_pace
  (steve_speed : ℝ := 3.8)
  (initial_gap : ℝ := 15)
  (finish_gap : ℝ := 2)
  (time : ℝ := 42.5)
  (steve_covered : ℝ := steve_speed * time)
  (john_covered : ℝ := steve_covered + initial_gap + finish_gap)
  (johns_pace_equation : J * time = john_covered) :
  J = 4.188 :=
by
  sorry

end NUMINAMATH_GPT_johns_pace_l1295_129592


namespace NUMINAMATH_GPT_fraction_comparison_l1295_129586

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l1295_129586


namespace NUMINAMATH_GPT_units_digit_27_64_l1295_129560

/-- 
  Given that the units digit of 27 is 7, 
  and the units digit of 64 is 4, 
  prove that the units digit of 27 * 64 is 8.
-/
theorem units_digit_27_64 : 
  ∀ (n m : ℕ), 
  (n % 10 = 7) → 
  (m % 10 = 4) → 
  ((n * m) % 10 = 8) :=
by
  intros n m h1 h2
  -- Utilize modular arithmetic properties
  sorry

end NUMINAMATH_GPT_units_digit_27_64_l1295_129560


namespace NUMINAMATH_GPT_ratio_of_triangle_areas_l1295_129549

theorem ratio_of_triangle_areas (kx ky k : ℝ)
(n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let A := (1 / 2) * (ky / m) * (kx / 2)
  let B := (1 / 2) * (kx / n) * (ky / 2)
  (A / B) = (n / m) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_areas_l1295_129549


namespace NUMINAMATH_GPT_length_of_longer_leg_of_smallest_triangle_l1295_129501

theorem length_of_longer_leg_of_smallest_triangle 
  (hypotenuse_largest : ℝ) 
  (h1 : hypotenuse_largest = 10)
  (h45 : ∀ hyp, (hyp / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2) = hypotenuse_largest / 4) :
  (hypotenuse_largest / 4) = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_length_of_longer_leg_of_smallest_triangle_l1295_129501


namespace NUMINAMATH_GPT_fencing_required_l1295_129569

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (hL : L = 20) (hA : A = 80) (hW : A = L * W) :
  (L + 2 * W) = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_fencing_required_l1295_129569


namespace NUMINAMATH_GPT_average_first_six_numbers_l1295_129561

theorem average_first_six_numbers (A : ℝ) (h1 : (11 : ℝ) * 9.9 = (6 * A + 6 * 11.4 - 22.5)) : A = 10.5 :=
by sorry

end NUMINAMATH_GPT_average_first_six_numbers_l1295_129561


namespace NUMINAMATH_GPT_double_windows_downstairs_eq_twelve_l1295_129547

theorem double_windows_downstairs_eq_twelve
  (D : ℕ)
  (H1 : ∀ d, d = D → 4 * d + 32 = 80) :
  D = 12 :=
by
  sorry

end NUMINAMATH_GPT_double_windows_downstairs_eq_twelve_l1295_129547


namespace NUMINAMATH_GPT_value_of_fourth_set_l1295_129542

def value_in_set (a b c d : ℕ) : ℕ :=
  (a * b * c * d) - (a + b + c + d)

theorem value_of_fourth_set :
  value_in_set 1 5 6 7 = 191 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fourth_set_l1295_129542


namespace NUMINAMATH_GPT_original_square_area_is_144_square_centimeters_l1295_129513

noncomputable def area_of_original_square (x : ℝ) : ℝ :=
  x^2 - (x - 3) * (x - 5)

theorem original_square_area_is_144_square_centimeters (x : ℝ) (h : area_of_original_square x = 81) :
  (x = 12) → (x^2 = 144) :=
by
  sorry

end NUMINAMATH_GPT_original_square_area_is_144_square_centimeters_l1295_129513


namespace NUMINAMATH_GPT_nalani_net_amount_l1295_129584

-- Definitions based on the conditions
def luna_birth := 10 -- Luna gave birth to 10 puppies
def stella_birth := 14 -- Stella gave birth to 14 puppies
def luna_sold := 8 -- Nalani sold 8 puppies from Luna's litter
def stella_sold := 10 -- Nalani sold 10 puppies from Stella's litter
def luna_price := 200 -- Price per puppy for Luna's litter is $200
def stella_price := 250 -- Price per puppy for Stella's litter is $250
def luna_cost := 80 -- Cost of raising each puppy from Luna's litter is $80
def stella_cost := 90 -- Cost of raising each puppy from Stella's litter is $90

-- Theorem stating the net amount received by Nalani
theorem nalani_net_amount : 
        luna_sold * luna_price + stella_sold * stella_price - 
        (luna_birth * luna_cost + stella_birth * stella_cost) = 2040 :=
by 
  sorry

end NUMINAMATH_GPT_nalani_net_amount_l1295_129584


namespace NUMINAMATH_GPT_mark_charged_more_hours_l1295_129597

theorem mark_charged_more_hours (P K M : ℕ) 
  (h_total : P + K + M = 144)
  (h_pat_kate : P = 2 * K)
  (h_pat_mark : P = M / 3) : M - K = 80 := 
by
  sorry

end NUMINAMATH_GPT_mark_charged_more_hours_l1295_129597


namespace NUMINAMATH_GPT_remainder_when_three_times_number_minus_seven_divided_by_seven_l1295_129571

theorem remainder_when_three_times_number_minus_seven_divided_by_seven (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_when_three_times_number_minus_seven_divided_by_seven_l1295_129571


namespace NUMINAMATH_GPT_smallest_sum_Q_lt_7_9_l1295_129508

def Q (N k : ℕ) : ℚ := (N + 1) / (N + k + 1)

theorem smallest_sum_Q_lt_7_9 : 
    ∃ N k : ℕ, (N + k) % 4 = 0 ∧ Q N k < 7 / 9 ∧ (∀ N' k' : ℕ, (N' + k') % 4 = 0 ∧ Q N' k' < 7 / 9 → N' + k' ≥ N + k) ∧ N + k = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_Q_lt_7_9_l1295_129508


namespace NUMINAMATH_GPT_positive_difference_perimeters_l1295_129566

theorem positive_difference_perimeters (length width : ℝ) 
    (cut_rectangles : ℕ) 
    (H : length = 6 ∧ width = 9 ∧ cut_rectangles = 4) : 
    ∃ (p1 p2 : ℝ), (p1 = 24 ∧ p2 = 15) ∧ (abs (p1 - p2) = 9) :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_perimeters_l1295_129566


namespace NUMINAMATH_GPT_product_of_two_numbers_l1295_129576

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 120) :
  x * y = -20 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1295_129576


namespace NUMINAMATH_GPT_number_of_triangles_l1295_129588

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l1295_129588


namespace NUMINAMATH_GPT_greatest_groups_of_stuffed_animals_l1295_129512

def stuffed_animals_grouping : Prop :=
  let cats := 26
  let dogs := 14
  let bears := 18
  let giraffes := 22
  gcd (gcd (gcd cats dogs) bears) giraffes = 2

theorem greatest_groups_of_stuffed_animals : stuffed_animals_grouping :=
by sorry

end NUMINAMATH_GPT_greatest_groups_of_stuffed_animals_l1295_129512


namespace NUMINAMATH_GPT_avg_speed_trip_l1295_129514

noncomputable def distance_travelled (speed time : ℕ) : ℕ := speed * time

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ := total_distance / total_time

theorem avg_speed_trip :
  let first_leg_speed := 75
  let first_leg_time := 4
  let second_leg_speed := 60
  let second_leg_time := 2
  let total_time := first_leg_time + second_leg_time
  let first_leg_distance := distance_travelled first_leg_speed first_leg_time
  let second_leg_distance := distance_travelled second_leg_speed second_leg_time
  let total_distance := first_leg_distance + second_leg_distance
  average_speed total_distance total_time = 70 :=
by
  sorry

end NUMINAMATH_GPT_avg_speed_trip_l1295_129514


namespace NUMINAMATH_GPT_minimum_value_expr_l1295_129568

noncomputable def expr (x y z : ℝ) : ℝ := 
  3 * x^2 + 2 * x * y + 3 * y^2 + 2 * y * z + 3 * z^2 - 3 * x + 3 * y - 3 * z + 9

theorem minimum_value_expr : 
  ∃ (x y z : ℝ), ∀ (a b c : ℝ), expr a b c ≥ expr x y z ∧ expr x y z = 3/2 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_l1295_129568


namespace NUMINAMATH_GPT_combined_rainfall_is_23_l1295_129528

-- Define the conditions
def monday_hours : ℕ := 7
def monday_rate : ℕ := 1
def tuesday_hours : ℕ := 4
def tuesday_rate : ℕ := 2
def wednesday_hours : ℕ := 2
def wednesday_rate (tuesday_rate : ℕ) : ℕ := 2 * tuesday_rate

-- Calculate the rainfalls
def monday_rainfall : ℕ := monday_hours * monday_rate
def tuesday_rainfall : ℕ := tuesday_hours * tuesday_rate
def wednesday_rainfall (wednesday_rate : ℕ) : ℕ := wednesday_hours * wednesday_rate

-- Define the total rainfall
def total_rainfall : ℕ :=
  monday_rainfall + tuesday_rainfall + wednesday_rainfall (wednesday_rate tuesday_rate)

theorem combined_rainfall_is_23 : total_rainfall = 23 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_combined_rainfall_is_23_l1295_129528


namespace NUMINAMATH_GPT_range_of_b_l1295_129509

theorem range_of_b (b : ℝ) :
  (∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 
    y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔ 
    1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1295_129509
