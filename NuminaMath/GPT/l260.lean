import Mathlib

namespace max_value_fourth_power_l260_260044

theorem max_value_fourth_power (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) : 
  a^4 + b^4 + c^4 + d^4 ≤ 4^(4/3) :=
sorry

end max_value_fourth_power_l260_260044


namespace driving_time_ratio_l260_260462

theorem driving_time_ratio
  (t28 t60 : ℚ) -- time driving at 28 mph and 60 mph in hours
  (total_driving_time : ℚ) (total_distance : ℚ)
  (t_bike : ℚ) (bike_speed : ℚ) :
  t28 + t60 = total_driving_time →
  total_distance = (28 * t28 + 60 * t60) →
  total_distance = bike_speed * t_bike →
  total_driving_time = 0.5 → -- converting 30 minutes to hours
  t_bike = 2 → -- Jake bikes for 2 hours
  bike_speed = 11 → -- Jake's biking speed is 11 miles per hour
  (t28 / total_driving_time) = 1 / 2 :=
begin
  sorry
end

end driving_time_ratio_l260_260462


namespace ann_frosting_cakes_l260_260936

theorem ann_frosting_cakes (normalRate sprainedRate cakes : ℕ) (H1 : normalRate = 5) (H2 : sprainedRate = 8) (H3 : cakes = 10) :
  (sprainedRate * cakes) - (normalRate * cakes) = 30 :=
by
  -- Substitute the provided values into the expression
  rw [H1, H2, H3]
  -- Evaluate the expression
  norm_num

end ann_frosting_cakes_l260_260936


namespace solve_for_x_and_compute_value_l260_260197

theorem solve_for_x_and_compute_value (x : ℝ) (h : 5 * x - 3 = 15 * x + 15) : 6 * (x + 5) = 19.2 := by
  sorry

end solve_for_x_and_compute_value_l260_260197


namespace overall_gain_percentage_l260_260105

def cost_of_A : ℝ := 100
def selling_price_of_A : ℝ := 125
def cost_of_B : ℝ := 200
def selling_price_of_B : ℝ := 250
def cost_of_C : ℝ := 150
def selling_price_of_C : ℝ := 180

theorem overall_gain_percentage :
  ((selling_price_of_A + selling_price_of_B + selling_price_of_C) - (cost_of_A + cost_of_B + cost_of_C)) / (cost_of_A + cost_of_B + cost_of_C) * 100 = 23.33 := 
by
  sorry

end overall_gain_percentage_l260_260105


namespace two_times_six_pow_n_plus_one_ne_product_of_consecutive_l260_260596

theorem two_times_six_pow_n_plus_one_ne_product_of_consecutive (n k : ℕ) :
  2 * (6 ^ n + 1) ≠ k * (k + 1) :=
sorry

end two_times_six_pow_n_plus_one_ne_product_of_consecutive_l260_260596


namespace probability_even_first_odd_second_l260_260932

-- Definitions based on the conditions
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := {2, 4, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Probability calculations
def prob_even := (even_numbers.card : ℚ) / (die_sides.card : ℚ)
def prob_odd := (odd_numbers.card : ℚ) / (die_sides.card : ℚ)

-- Proof statement
theorem probability_even_first_odd_second :
  prob_even * prob_odd = 1 / 4 :=
by
  sorry

end probability_even_first_odd_second_l260_260932


namespace landscape_length_l260_260205

theorem landscape_length (b : ℝ) 
  (h1 : ∀ (l : ℝ), l = 8 * b) 
  (A : ℝ)
  (h2 : A = 8 * b^2)
  (Playground_area : ℝ)
  (h3 : Playground_area = 1200)
  (h4 : Playground_area = (1 / 6) * A) :
  ∃ (l : ℝ), l = 240 :=
by 
  sorry

end landscape_length_l260_260205


namespace species_population_estimate_l260_260918

theorem species_population_estimate :
  ∃ (N_A N_B N_C : ℕ), 
    (40 / 2400 = 3 / 180) ∧
    (40 / 1440 = 5 / 180) ∧
    (40 / 3600 = 2 / 180) ∧
    N_A = 2400 ∧
    N_B = 1440 ∧
    N_C = 3600 :=
by {
  existsi (2400 : ℕ),
  existsi (1440 : ℕ),
  existsi (3600 : ℕ),
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { reflexivity },
  split,
  { reflexivity },
  { reflexivity }
}

end species_population_estimate_l260_260918


namespace sample_size_l260_260522

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ) (elderly_employees : ℕ) (young_in_sample : ℕ)

theorem sample_size (h1 : total_employees = 750) (h2 : young_employees = 350) (h3 : middle_aged_employees = 250) (h4 : elderly_employees = 150) (h5 : young_in_sample = 7) :
  ∃ sample_size, young_in_sample * total_employees / young_employees = sample_size ∧ sample_size = 15 :=
by
  sorry

end sample_size_l260_260522


namespace number_of_subsets_of_set_l260_260368

theorem number_of_subsets_of_set {n : ℕ} (h : n = 2016) :
  (2^2016) = 2^2016 :=
by
  sorry

end number_of_subsets_of_set_l260_260368


namespace distinct_lines_isosceles_not_equilateral_l260_260367

-- Define a structure for an isosceles triangle that is not equilateral
structure IsoscelesButNotEquilateralTriangle :=
  (a b c : ℕ)    -- sides of the triangle
  (h₁ : a = b)   -- two equal sides
  (h₂ : a ≠ c)   -- not equilateral (not all three sides are equal)

-- Define that the number of distinct lines representing altitudes, medians, and interior angle bisectors is 5
theorem distinct_lines_isosceles_not_equilateral (T : IsoscelesButNotEquilateralTriangle) : 
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end distinct_lines_isosceles_not_equilateral_l260_260367


namespace courses_students_problem_l260_260806

theorem courses_students_problem :
  let courses := Fin 6 -- represent 6 courses
  let students := Fin 20 -- represent 20 students
  (∀ (C C' : courses), ∀ (S : Finset students), S.card = 5 → 
    ¬ ((∀ s ∈ S, ∃ s_courses : Finset courses, C ∈ s_courses ∧ C' ∈ s_courses) ∨ 
       (∀ s ∈ S, ∃ s_courses : Finset courses, C ∉ s_courses ∧ C' ∉ s_courses))) :=
by sorry

end courses_students_problem_l260_260806


namespace apples_in_pile_l260_260497

-- Define the initial number of apples in the pile
def initial_apples : ℕ := 8

-- Define the number of added apples
def added_apples : ℕ := 5

-- Define the total number of apples
def total_apples : ℕ := initial_apples + added_apples

-- Prove that the total number of apples is 13
theorem apples_in_pile : total_apples = 13 :=
by
  sorry

end apples_in_pile_l260_260497


namespace olivia_card_value_l260_260319

theorem olivia_card_value (x : ℝ) (hx1 : 90 < x ∧ x < 180)
  (h_sin_pos : Real.sin x > 0) (h_cos_neg : Real.cos x < 0) (h_tan_neg : Real.tan x < 0)
  (h_olivia_distinguish : ∀ (a b c : ℝ), 
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (∃! a, a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x)) :
  Real.sin 135 = Real.cos 45 := 
sorry

end olivia_card_value_l260_260319


namespace three_digit_numbers_with_square_ending_in_them_l260_260866

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l260_260866


namespace ironed_clothing_l260_260781

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l260_260781


namespace password_decryption_prob_l260_260814

theorem password_decryption_prob :
  let p1 : ℚ := 1 / 5
  let p2 : ℚ := 1 / 4
  let q1 : ℚ := 1 - p1
  let q2 : ℚ := 1 - p2
  (1 - (q1 * q2)) = 2 / 5 :=
by
  sorry

end password_decryption_prob_l260_260814


namespace total_board_length_l260_260827

-- Defining the lengths of the pieces of the board
def shorter_piece_length : ℕ := 23
def longer_piece_length : ℕ := 2 * shorter_piece_length

-- Stating the theorem that the total length of the board is 69 inches
theorem total_board_length : shorter_piece_length + longer_piece_length = 69 :=
by
  -- The proof is omitted for now
  sorry

end total_board_length_l260_260827


namespace find_covered_number_l260_260919

theorem find_covered_number (a x : ℤ) (h : (x - a) / 2 = x + 3) (hx : x = -7) : a = 1 := by
  sorry

end find_covered_number_l260_260919


namespace find_time_for_compound_interest_l260_260389

noncomputable def compound_interest_time 
  (A P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem find_time_for_compound_interest :
  compound_interest_time 500 453.51473922902494 0.05 1 = 2 :=
sorry

end find_time_for_compound_interest_l260_260389


namespace mrs_jane_total_coins_l260_260112

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l260_260112


namespace number_of_ways_to_distribute_balls_l260_260567

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l260_260567


namespace probability_triangle_l260_260309

noncomputable def points : List (ℕ × ℕ) := [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2), (3, 3)]

def collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def is_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop := ¬ collinear p1 p2 p3

axiom collinear_ACEF : collinear (0, 0) (1, 1) (2, 2) ∧ collinear (0, 0) (1, 1) (3, 3) ∧ collinear (1, 1) (2, 2) (3, 3)
axiom collinear_BCD : collinear (2, 0) (1, 1) (0, 2)

theorem probability_triangle : 
  let total := 20
  let collinear_ACEF := 4
  let collinear_BCD := 1
  (total - collinear_ACEF - collinear_BCD) / total = 3 / 4 :=
by
  sorry

end probability_triangle_l260_260309


namespace evaluate_expression_l260_260283

variable (m n p : ℝ)

theorem evaluate_expression 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := 
by 
  sorry

end evaluate_expression_l260_260283


namespace average_speed_l260_260976

-- Definitions of conditions
def speed_first_hour : ℝ := 120
def speed_second_hour : ℝ := 60
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := 2

-- Theorem stating the equivalent proof problem
theorem average_speed : total_distance / total_time = 90 := by
  sorry

end average_speed_l260_260976


namespace evaluate_root_power_l260_260416

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l260_260416


namespace children_to_add_l260_260594

def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def desired_children := 30

theorem children_to_add : (desired_children - children) = 10 := by
  sorry

end children_to_add_l260_260594


namespace harkamal_mangoes_l260_260159

theorem harkamal_mangoes (m : ℕ) (h1: 8 * 70 = 560) (h2 : m * 50 + 560 = 1010) : m = 9 :=
by
  sorry

end harkamal_mangoes_l260_260159


namespace quadratic_term_free_polynomial_l260_260452

theorem quadratic_term_free_polynomial (m : ℤ) (h : 36 + 12 * m = 0) : m^3 = -27 := by
  -- Proof goes here
  sorry

end quadratic_term_free_polynomial_l260_260452


namespace smallest_n_45_l260_260629

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260629


namespace mechanic_hours_l260_260526

theorem mechanic_hours (h : ℕ) (labor_cost_per_hour parts_cost total_bill : ℕ) 
  (H1 : labor_cost_per_hour = 45) 
  (H2 : parts_cost = 225) 
  (H3 : total_bill = 450) 
  (H4 : labor_cost_per_hour * h + parts_cost = total_bill) : 
  h = 5 := 
by
  sorry

end mechanic_hours_l260_260526


namespace determinant_expr_l260_260471

theorem determinant_expr (a b c p q r : ℝ) 
  (h1 : ∀ x, Polynomial.eval x (Polynomial.C a * Polynomial.C b * Polynomial.C c - Polynomial.C p * (Polynomial.C a * Polynomial.C b + Polynomial.C b * Polynomial.C c + Polynomial.C c * Polynomial.C a) + Polynomial.C q * (Polynomial.C a + Polynomial.C b + Polynomial.C c) - Polynomial.C r) = 0) :
  Matrix.det ![
    ![2 + a, 1, 1],
    ![1, 2 + b, 1],
    ![1, 1, 2 + c]
  ] = r + 2*q + 4*p + 4 :=
sorry

end determinant_expr_l260_260471


namespace factor_expression_l260_260417

theorem factor_expression (x y : ℤ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end factor_expression_l260_260417


namespace total_clothing_ironed_l260_260787

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l260_260787


namespace range_of_b_over_a_l260_260141

noncomputable def f (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def f' (a b x : ℝ) : ℝ := (b / x^2 + a * x - b / x - a) * Real.exp x

theorem range_of_b_over_a (a b : ℝ) (h₀ : a > 0) (h₁ : ∃ x : ℝ, 1 < x ∧ f a b x + f' a b x = 0) : 
  -1 < b / a := sorry

end range_of_b_over_a_l260_260141


namespace determine_transportation_mode_l260_260258

def distance : ℝ := 60 -- in kilometers
def time : ℝ := 3 -- in hours
def speed_of_walking : ℝ := 5 -- typical speed in km/h
def speed_of_bicycle_riding : ℝ := 15 -- lower bound of bicycle speed in km/h
def speed_of_driving_a_car : ℝ := 20 -- typical minimum speed in km/h

theorem determine_transportation_mode : (distance / time) = speed_of_driving_a_car ∧ speed_of_driving_a_car ≥ speed_of_walking + speed_of_bicycle_riding - speed_of_driving_a_car := sorry

end determine_transportation_mode_l260_260258


namespace minimum_value_of_x_plus_y_existence_of_minimum_value_l260_260281

theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + 2 * x + y = 8) :
  x + y ≥ 2 * Real.sqrt 10 - 3 :=
sorry

theorem existence_of_minimum_value (x y : ℝ) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 8 ∧ x + y = 2 * Real.sqrt 10 - 3 :=
sorry

end minimum_value_of_x_plus_y_existence_of_minimum_value_l260_260281


namespace solve_cubic_root_eq_l260_260861

theorem solve_cubic_root_eq (x : ℝ) : (∃ x, 3 - x / 3 = -8) -> x = 33 :=
by
  sorry

end solve_cubic_root_eq_l260_260861


namespace minimum_inlets_needed_l260_260529

noncomputable def waterInflow (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := x * a - b

theorem minimum_inlets_needed (a b : ℝ) (ha : a = b)
  (h1 : (4 * a - b) * 5 = (2 * a - b) * 15)
  (h2 : (a * 9 - b) * 2 ≥ 1) : 
  ∃ n : ℕ, 2 * (a * n - b) ≥ (4 * a - b) * 5 := 
by 
  sorry

end minimum_inlets_needed_l260_260529


namespace range_m_l260_260722

namespace MathProof

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_m
  (m : ℝ)
  (h : m > 0)
  (a b c : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 2)
  (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_triangle : f a m ^ 2 + f b m ^ 2 = f c m ^ 2 ∨
                f a m ^ 2 + f c m ^ 2 = f b m ^ 2 ∨
                f b m ^ 2 + f c m ^ 2 = f a m ^ 2) :
  0 < m ∧ m < 3 + 4 * Real.sqrt 2 :=
by
  sorry

end MathProof

end range_m_l260_260722


namespace evaluate_root_power_l260_260415

theorem evaluate_root_power : (real.root 4 16)^12 = 4096 := by
  sorry

end evaluate_root_power_l260_260415


namespace negate_exists_implies_forall_l260_260078

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l260_260078


namespace solution_to_problem_l260_260300

theorem solution_to_problem (x : ℝ) (h : 12^(Real.log 7 / Real.log 12) = 10 * x + 3) : x = 2 / 5 :=
by sorry

end solution_to_problem_l260_260300


namespace solve_problem_l260_260981

-- Define the polynomial p(x)
noncomputable def p (x : ℂ) : ℂ := x^2 - x + 1

-- Define the root condition
def is_root (α : ℂ) : Prop := p (p (p (p α))) = 0

-- Define the expression to evaluate
noncomputable def expression (α : ℂ) : ℂ := (p α - 1) * p α * p (p α) * p (p (p α))

-- State the theorem asserting the required equality
theorem solve_problem (α : ℂ) (hα : is_root α) : expression α = -1 :=
sorry

end solve_problem_l260_260981


namespace proof_problem_l260_260749

variable (p q : Prop)

theorem proof_problem
  (h₁ : p ∨ q)
  (h₂ : ¬p) :
  ¬p ∧ q :=
by
  sorry

end proof_problem_l260_260749


namespace total_apples_l260_260498

-- Definitions and Conditions
variable (a : ℕ) -- original number of apples in the first pile (scaled integer type)
variable (n m : ℕ) -- arbitrary positions in the sequence

-- Arithmetic sequence of initial piles
def initial_piles := [a, 2*a, 3*a, 4*a, 5*a, 6*a]

-- Given condition transformations
def after_removal_distribution (initial_piles : List ℕ) (k : ℕ) : List ℕ :=
  match k with
  | 0 => [0, 2*a + 10, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 1 => [a + 10, 0, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 2 => [a + 10, 2*a + 20, 0, 4*a + 30, 5*a + 40, 6*a + 50]
  | 3 => [a + 10, 2*a + 20, 3*a + 30, 0, 5*a + 40, 6*a + 50]
  | 4 => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 0, 6*a + 50]
  | _ => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 5*a + 50, 0]

-- Prove the total number of apples
theorem total_apples : (a = 35) → (a + 2 * a + 3 * a + 4 * a + 5 * a + 6 * a = 735) :=
by
  intros h1
  sorry

end total_apples_l260_260498


namespace arithmetic_sequence_common_difference_l260_260755

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ)
  (h_a4 : a₁ + 3 * d = -2)
  (h_sum : 10 * a₁ + 45 * d = 65) :
  d = 17 / 3 :=
sorry

end arithmetic_sequence_common_difference_l260_260755


namespace M_minus_N_positive_l260_260005

variable (a b : ℝ)

def M : ℝ := 10 * a^2 + b^2 - 7 * a + 8
def N : ℝ := a^2 + b^2 + 5 * a + 1

theorem M_minus_N_positive : M a b - N a b ≥ 3 := by
  sorry

end M_minus_N_positive_l260_260005


namespace smallest_n_satisfies_conditions_l260_260651

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260651


namespace James_selling_percentage_l260_260314

def James_selling_percentage_proof : Prop :=
  ∀ (total_cost original_price return_cost extra_item bought_price out_of_pocket sold_amount : ℝ),
    total_cost = 3000 →
    return_cost = 700 + 500 →
    extra_item = 500 * 1.2 →
    bought_price = 100 →
    out_of_pocket = 2020 →
    sold_amount = out_of_pocket - (total_cost - return_cost + bought_price) →
    sold_amount / extra_item * 100 = 20

theorem James_selling_percentage : James_selling_percentage_proof :=
by
  sorry

end James_selling_percentage_l260_260314


namespace radius_of_circle_B_l260_260256

theorem radius_of_circle_B (r_A r_D : ℝ) (r_B : ℝ) (hA : r_A = 2) (hD : r_D = 4) 
  (congruent_BC : r_B = r_B) (tangent_condition : true) -- placeholder conditions
  (center_pass : true) -- placeholder conditions
  : r_B = (4 / 3) * (Real.sqrt 7 - 1) :=
sorry

end radius_of_circle_B_l260_260256


namespace sufficient_but_not_necessary_l260_260586

-- Definitions for lines a and b, and planes alpha and beta
variables {a b : Type} {α β : Type}

-- predicate for line a being in plane α
def line_in_plane (a : Type) (α : Type) : Prop := sorry

-- predicate for line b being perpendicular to plane β
def line_perpendicular_plane (b : Type) (β : Type) : Prop := sorry

-- predicate for plane α being parallel to plane β
def plane_parallel_plane (α : Type) (β : Type) : Prop := sorry

-- predicate for line a being perpendicular to line b
def line_perpendicular_line (a : Type) (b : Type) : Prop := sorry

-- Proof of the statement: The condition of line a being in plane α, line b being perpendicular to plane β,
-- and plane α being parallel to plane β is sufficient but not necessary for line a being perpendicular to line b.
theorem sufficient_but_not_necessary
  (a b : Type) (α β : Type)
  (h1 : line_in_plane a α)
  (h2 : line_perpendicular_plane b β)
  (h3 : plane_parallel_plane α β) :
  line_perpendicular_line a b :=
sorry

end sufficient_but_not_necessary_l260_260586


namespace sequence_a3_equals_1_over_3_l260_260167

theorem sequence_a3_equals_1_over_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1) + 1)) : 
  a 3 = 1 / 3 :=
sorry

end sequence_a3_equals_1_over_3_l260_260167


namespace boys_neither_happy_nor_sad_correct_l260_260229

def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def total_boys : ℕ := 16
def total_girls : ℕ := 44
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- The number of boys who are neither happy nor sad
def boys_neither_happy_nor_sad : ℕ :=
  total_boys - happy_boys - (sad_children - sad_girls)

theorem boys_neither_happy_nor_sad_correct : boys_neither_happy_nor_sad = 4 := by
  sorry

end boys_neither_happy_nor_sad_correct_l260_260229


namespace mike_spent_total_l260_260344

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l260_260344


namespace smallest_n_l260_260653

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260653


namespace find_special_three_digit_numbers_l260_260868

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l260_260868


namespace distance_covered_l260_260777

theorem distance_covered (t : ℝ) (s_kmph : ℝ) (distance : ℝ) (h1 : t = 180) (h2 : s_kmph = 18) : 
  distance = 900 :=
by 
  sorry

end distance_covered_l260_260777


namespace smallest_y_value_smallest_y_value_is_neg6_l260_260659

theorem smallest_y_value :
  ∀ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) → (y = -3 ∨ y = -6) :=
by
  sorry

theorem smallest_y_value_is_neg6 :
  ∃ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧ (y = -6) :=
by
  have H := smallest_y_value
  sorry

end smallest_y_value_smallest_y_value_is_neg6_l260_260659


namespace Elmo_books_count_l260_260129

-- Define the number of books each person has
def Stu_books : ℕ := 4
def Laura_books : ℕ := 2 * Stu_books
def Elmo_books : ℕ := 3 * Laura_books

-- The theorem we need to prove
theorem Elmo_books_count : Elmo_books = 24 := by
  -- this part is skipped since no proof is required
  sorry

end Elmo_books_count_l260_260129


namespace weight_difference_l260_260927

def weight_chemistry : ℝ := 7.12
def weight_geometry : ℝ := 0.62

theorem weight_difference : weight_chemistry - weight_geometry = 6.50 :=
by
  sorry

end weight_difference_l260_260927


namespace mike_spent_l260_260346

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l260_260346


namespace smallest_n_45_l260_260630

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260630


namespace a_seq_gt_one_l260_260930

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 + a
  else (1 / a_seq a (n - 1)) + a

theorem a_seq_gt_one (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ n : ℕ, 1 < a_seq a n :=
by {
  sorry
}

end a_seq_gt_one_l260_260930


namespace alloy_mixing_l260_260983

theorem alloy_mixing (x : ℕ) :
  (2 / 5) * 60 + (1 / 5) * x = 44 → x = 100 :=
by
  intros h1
  sorry

end alloy_mixing_l260_260983


namespace find_f6_l260_260607

variable {R : Type} [LinearOrderedField R]

def f : R → R := sorry

theorem find_f6 (h1 : ∀ x y : R, f (x - y) = f x * f y) (h2 : ∀ x : R, f x ≠ 0) : f 6 = 1 :=
sorry

end find_f6_l260_260607


namespace range_of_m_l260_260732

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) / x

theorem range_of_m (a m : ℝ) (h₀ : a > 0) (h₁ : f a (m^2 + 1) > f a (m^2 - m + 3)) 
  : m > 2 :=
sorry

end range_of_m_l260_260732


namespace required_decrease_l260_260680

noncomputable def price_after_increases (P : ℝ) : ℝ :=
  let P1 := 1.20 * P
  let P2 := 1.10 * P1
  1.15 * P2

noncomputable def price_after_discount (P : ℝ) : ℝ :=
  0.95 * price_after_increases P

noncomputable def price_after_tax (P : ℝ) : ℝ :=
  1.07 * price_after_discount P

theorem required_decrease (P : ℝ) (D : ℝ) : 
  (1 - D / 100) * price_after_tax P = P ↔ D = 35.1852 :=
by
  sorry

end required_decrease_l260_260680


namespace find_three_digit_numbers_l260_260864

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l260_260864


namespace find_M_value_l260_260081

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l260_260081


namespace triangle_ratio_inequality_l260_260727

/-- Given a triangle ABC, R is the radius of the circumscribed circle, 
    r is the radius of the inscribed circle, a is the length of the longest side,
    and h is the length of the shortest altitude. Prove that R / r > a / h. -/
theorem triangle_ratio_inequality
  (ABC : Triangle) (R r a h : ℝ)
  (hR : 2 * R ≥ a)
  (hr : 2 * r < h) :
  (R / r) > (a / h) :=
by
  -- sorry is used to skip the proof
  sorry

end triangle_ratio_inequality_l260_260727


namespace tan_expression_l260_260006

variables {γ δ : ℝ}

theorem tan_expression 
  (hγ : Real.tan γ = 3) 
  (hδ : Real.tan δ = 2) : 
  Real.tan (2 * γ - δ) = 11 / 2 := 
sorry

end tan_expression_l260_260006


namespace point_on_hyperbola_probability_l260_260057

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l260_260057


namespace clara_total_cookies_l260_260404

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l260_260404


namespace find_positive_value_of_X_l260_260043

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l260_260043


namespace min_fraction_value_l260_260145

theorem min_fraction_value 
    (a : ℕ → ℝ) 
    (S : ℕ → ℝ) 
    (d : ℝ) 
    (n : ℕ) 
    (h1 : ∀ {n}, a n = 5 + (n - 1) * d)
    (h2 : (a 2) * (a 10) = (a 4 - 1)^2) 
    (h3 : S n = (n * (a 1 + a n)) / 2)
    (h4 : a 1 = 5)
    (h5 : d > 0) :
    2 * S n + n + 32 ≥ (20 / 3) * (a n + 1) := sorry

end min_fraction_value_l260_260145


namespace probability_of_total_sum_of_dice_is_odd_l260_260964

-- Define the fair coin and fair dice rolls
def fair_coin : Pmf Bool :=
  Pmf.ofMultiset { True, False }.toMultiset

def fair_die : Pmf ℕ :=
  Pmf.ofMultiset { 1, 2, 3, 4, 5, 6 }.toMultiset

-- Define the problem conditions
def three_fair_coins_tossed_once : Pmf (List Bool) :=
  Pmf.replicateM 3 fair_coin

def heads_resulting_to_dice_rolls (heads : ℕ) : Pmf (List ℕ) :=
  Pmf.replicateM (2 * heads) fair_die

-- Define the problem statement and the probability function
def probability_odd_sum_of_dice : Pmf (List ℕ) → ℚ :=
  λ l, if l.sum % 2 = 1 then 1 else 0

noncomputable def total_probability_odd_sum : ℚ :=
  (∑ outcome in (three_fair_coins_tossed_once.support), 
  (three_fair_coins_tossed_once outcome) * 
  (let heads := outcome.count id in 
  ∑ rolls in (heads_resulting_to_dice_rolls heads).support,
  (heads_resulting_to_dice_rolls heads rolls) * 
  (probability_odd_sum_of_dice rolls)))

-- Proof to show the probability of odd sum is 7/16
theorem probability_of_total_sum_of_dice_is_odd : 
  total_probability_odd_sum = 7 / 16 :=
sorry

end probability_of_total_sum_of_dice_is_odd_l260_260964


namespace one_gt_one_others_lt_one_l260_260800

theorem one_gt_one_others_lt_one 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_prod : a * b * c = 1)
  (h_ineq : a + b + c > (1 / a) + (1 / b) + (1 / c)) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
sorry

end one_gt_one_others_lt_one_l260_260800


namespace units_digit_base7_product_l260_260049

theorem units_digit_base7_product (a b : ℕ) (ha : a = 354) (hb : b = 78) : (a * b) % 7 = 4 := by
  sorry

end units_digit_base7_product_l260_260049


namespace twenty_one_less_than_sixty_thousand_l260_260821

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 :=
by
  sorry

end twenty_one_less_than_sixty_thousand_l260_260821


namespace molecular_weight_of_Carbonic_acid_l260_260399

theorem molecular_weight_of_Carbonic_acid :
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  (H_atoms * H_weight + C_atoms * C_weight + O_atoms * O_weight) = 62.024 :=
by 
  let H_weight := 1.008
  let C_weight := 12.011
  let O_weight := 15.999
  let H_atoms := 2
  let C_atoms := 1
  let O_atoms := 3
  sorry

end molecular_weight_of_Carbonic_acid_l260_260399


namespace three_digit_numbers_with_square_ending_in_them_l260_260867

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l260_260867


namespace parabola_coeff_sum_l260_260858

theorem parabola_coeff_sum (p q r : ℝ) :
  (∀ x : ℝ, y = p * x ^ 2 + q * x + r ∧ ((0 : ℝ), -2) ∧ vertex (-3 : ℝ) (4) -  p + q + r = -20 / 3= Rational)atic := 
  sorry

end parabola_coeff_sum_l260_260858


namespace abs_neg_2023_l260_260793

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l260_260793


namespace smallest_positive_n_l260_260646

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260646


namespace heavy_rain_duration_l260_260598

-- Define the conditions as variables and constants
def initial_volume := 100 -- Initial volume in liters
def final_volume := 280   -- Final volume in liters
def flow_rate := 2        -- Flow rate in liters per minute

-- Define the duration query as a theorem to be proved
theorem heavy_rain_duration : 
  (final_volume - initial_volume) / flow_rate = 90 := 
by
  sorry

end heavy_rain_duration_l260_260598


namespace monomial_sum_l260_260085

theorem monomial_sum (m n : ℤ) (h1 : n - 1 = 4) (h2 : m - 1 = 2) : m - 2 * n = -7 := by
  sorry

end monomial_sum_l260_260085


namespace find_y_l260_260008

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l260_260008


namespace solve_for_x_l260_260747

theorem solve_for_x (x : ℝ) (h : 3*x - 4*x + 5*x = 140) : x = 35 :=
by 
  sorry

end solve_for_x_l260_260747


namespace remaining_days_to_finish_coke_l260_260002

def initial_coke_in_ml : ℕ := 2000
def daily_consumption_in_ml : ℕ := 200
def days_already_drunk : ℕ := 3

theorem remaining_days_to_finish_coke : 
  (initial_coke_in_ml / daily_consumption_in_ml) - days_already_drunk = 7 := 
by
  sorry -- Proof placeholder

end remaining_days_to_finish_coke_l260_260002


namespace parallelepiped_surface_area_l260_260995

theorem parallelepiped_surface_area (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 12) 
  (h2 : a * b * c = 8) : 
  6 * (a^2) = 24 :=
by
  sorry

end parallelepiped_surface_area_l260_260995


namespace original_costs_l260_260118

theorem original_costs (P_old P_second_oldest : ℝ) (h1 : 0.9 * P_old = 1800) (h2 : 0.85 * P_second_oldest = 900) :
  P_old + P_second_oldest = 3058.82 :=
by sorry

end original_costs_l260_260118


namespace least_number_of_table_entries_l260_260759

-- Given conditions
def num_towns : ℕ := 6

-- Theorem statement
theorem least_number_of_table_entries : (num_towns * (num_towns - 1)) / 2 = 15 := by
  -- Proof goes here.
  sorry

end least_number_of_table_entries_l260_260759


namespace find_y_l260_260009

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l260_260009


namespace field_day_difference_l260_260312

def class_students (girls boys : ℕ) := girls + boys

def grade_students 
  (class1_girls class1_boys class2_girls class2_boys class3_girls class3_boys : ℕ) :=
  (class1_girls + class2_girls + class3_girls, class1_boys + class2_boys + class3_boys)

def diff_students (g1 b1 g2 b2 g3 b3 : ℕ) := 
  b1 + b2 + b3 - (g1 + g2 + g3)

theorem field_day_difference :
  let g3_1 := 10   -- 3rd grade first class girls
  let b3_1 := 14   -- 3rd grade first class boys
  let g3_2 := 12   -- 3rd grade second class girls
  let b3_2 := 10   -- 3rd grade second class boys
  let g3_3 := 11   -- 3rd grade third class girls
  let b3_3 :=  9   -- 3rd grade third class boys
  let g4_1 := 12   -- 4th grade first class girls
  let b4_1 := 13   -- 4th grade first class boys
  let g4_2 := 15   -- 4th grade second class girls
  let b4_2 := 11   -- 4th grade second class boys
  let g4_3 := 14   -- 4th grade third class girls
  let b4_3 := 12   -- 4th grade third class boys
  let g5_1 :=  9   -- 5th grade first class girls
  let b5_1 := 13   -- 5th grade first class boys
  let g5_2 := 10   -- 5th grade second class girls
  let b5_2 := 11   -- 5th grade second class boys
  let g5_3 := 11   -- 5th grade third class girls
  let b5_3 := 14   -- 5th grade third class boys
  diff_students (g3_1 + g3_2 + g3_3 + g4_1 + g4_2 + g4_3 + g5_1 + g5_2 + g5_3)
                (b3_1 + b3_2 + b3_3 + b4_1 + b4_2 + b4_3 + b5_1 + b5_2 + b5_3) = 3 :=
by
  sorry

end field_day_difference_l260_260312


namespace length_of_train_l260_260999

variable (d_train d_bridge v t : ℝ)

theorem length_of_train
  (h1 : v = 12.5) 
  (h2 : t = 30) 
  (h3 : d_bridge = 255) 
  (h4 : v * t = d_train + d_bridge) : 
  d_train = 120 := 
by {
  -- We should infer from here that d_train = 120
  sorry
}

end length_of_train_l260_260999


namespace sequence_an_sum_sequence_Tn_l260_260143

theorem sequence_an (k c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = k * c ^ n - k) (ha2 : a 2 = 4) (ha6 : a 6 = 8 * a 3) :
  ∀ n, a n = 2 ^ n :=
by
  -- Proof is assumed to be given
  sorry

theorem sum_sequence_Tn (a : ℕ → ℝ) (T : ℕ → ℝ)
  (ha : ∀ n, a n = 2 ^ n) :
  ∀ n, T n = (n - 1) * 2 ^ (n + 1) + 2 :=
by
  -- Proof is assumed to be given
  sorry

end sequence_an_sum_sequence_Tn_l260_260143


namespace complex_division_l260_260007

theorem complex_division (i : ℂ) (h : i^2 = -1) : (2 + i) / (1 - 2 * i) = i := 
by
  sorry

end complex_division_l260_260007


namespace smallest_n_45_l260_260634

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260634


namespace find_M_value_l260_260080

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l260_260080


namespace divide_and_add_l260_260963

theorem divide_and_add (x : ℤ) (h1 : x = 95) : (x / 5) + 23 = 42 := by
  sorry

end divide_and_add_l260_260963


namespace smaller_angle_is_70_l260_260025

def measure_of_smaller_angle (x : ℕ) : Prop :=
  (x + (x + 40) = 180) ∧ (2 * x - 60 = 80)

theorem smaller_angle_is_70 {x : ℕ} : measure_of_smaller_angle x → x = 70 :=
by
  sorry

end smaller_angle_is_70_l260_260025


namespace marathon_time_l260_260525

theorem marathon_time (distance_first_10 : ℕ) (time_first_10 : ℕ) (total_distance : ℕ) (pace_reduction : ℚ) : 
  total_distance = 26 → 
  distance_first_10 = 10 → 
  time_first_10 = 1 →
  pace_reduction = 0.8 →
  ∃ total_time : ℚ, total_time = 3 :=
by 
  intros h_total_distance h_distance_first_10 h_time_first_10 h_pace_reduction
  let pace_first_10 := distance_first_10 / time_first_10
  let pace_reduced := pace_first_10 * pace_reduction
  let remaining_distance := total_distance - distance_first_10
  let remaining_time := remaining_distance / pace_reduced
  let total_time := time_first_10 + remaining_time
  use total_time
  rw [h_total_distance, h_distance_first_10, h_time_first_10, h_pace_reduction]
  norm_num
  sorry

end marathon_time_l260_260525


namespace exponent_property_l260_260139

theorem exponent_property (a x y : ℝ) (h1 : 0 < a) (h2 : a ^ x = 2) (h3 : a ^ y = 3) : a ^ (x - y) = 2 / 3 := 
by
  sorry

end exponent_property_l260_260139


namespace radiator_water_fraction_l260_260520

theorem radiator_water_fraction :
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  fraction_remaining_per_replacement^4 = 81 / 256 := by
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  sorry

end radiator_water_fraction_l260_260520


namespace arithmetic_sequence_condition_l260_260518

theorem arithmetic_sequence_condition (a : ℕ → ℝ) :
  (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2)) ↔
  (∀ n ∈ {k : ℕ | k > 0}, a (n+1) - a n = a (n+2) - a (n+1)) ∧ ¬ (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2) → a (n+1) = a n) :=
sorry

end arithmetic_sequence_condition_l260_260518


namespace example_inequality_l260_260182

variable (a b c : ℝ)

theorem example_inequality 
  (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
by
  sorry

end example_inequality_l260_260182


namespace triangle_bisector_ratio_l260_260168

theorem triangle_bisector_ratio (AB BC CA : ℝ) (h_AB_pos : 0 < AB) (h_BC_pos : 0 < BC) (h_CA_pos : 0 < CA)
  (AA1_bisector : True) (BB1_bisector : True) (O_intersection : True) : 
  AA1 / OA1 = 3 :=
by
  sorry

end triangle_bisector_ratio_l260_260168


namespace integer_squares_l260_260540

theorem integer_squares (x y : ℤ) 
  (hx : ∃ a : ℤ, x + y = a^2)
  (h2x3y : ∃ b : ℤ, 2 * x + 3 * y = b^2)
  (h3xy : ∃ c : ℤ, 3 * x + y = c^2) : 
  x = 0 ∧ y = 0 := 
by { sorry }

end integer_squares_l260_260540


namespace range_of_a_for_real_roots_l260_260020

theorem range_of_a_for_real_roots (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ), a*x^2 + 2*x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_for_real_roots_l260_260020


namespace general_term_correct_S_maximum_value_l260_260734

noncomputable def general_term (n : ℕ) : ℤ :=
  if n = 1 then -1 + 24 else (-n^2 + 24 * n) - (-(n - 1)^2 + 24 * (n - 1))

noncomputable def S (n : ℕ) : ℤ :=
  -n^2 + 24 * n

theorem general_term_correct (n : ℕ) (h : 1 ≤ n) : general_term n = -2 * n + 25 := by
  sorry

theorem S_maximum_value : ∃ n : ℕ, S n = 144 ∧ ∀ m : ℕ, S m ≤ 144 := by
  existsi 12
  sorry

end general_term_correct_S_maximum_value_l260_260734


namespace intersect_at_one_point_l260_260536

theorem intersect_at_one_point (a : ℝ) : 
  (a * (4 * 4) + 4 * 4 * 6 = 0) -> a = 2 / (3: ℝ) :=
by sorry

end intersect_at_one_point_l260_260536


namespace basic_computer_price_l260_260978

theorem basic_computer_price (C P : ℝ) 
(h1 : C + P = 2500) 
(h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end basic_computer_price_l260_260978


namespace four_letters_three_mailboxes_l260_260087

theorem four_letters_three_mailboxes : (3 ^ 4) = 81 :=
  by sorry

end four_letters_three_mailboxes_l260_260087


namespace correct_operation_l260_260661

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^4 ≠ a^6) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  ((a^2 * b)^3 = a^6 * b^3) ∧
  (a^6 / a^6 ≠ a) :=
by
  sorry

end correct_operation_l260_260661


namespace distance_covered_at_40_kmph_l260_260975

theorem distance_covered_at_40_kmph (x : ℝ) 
  (h₁ : x / 40 + (250 - x) / 60 = 5.5) :
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l260_260975


namespace pipes_fill_cistern_together_in_15_minutes_l260_260621

-- Define the problem's conditions in Lean
def PipeA_rate := (1 / 2) / 15
def PipeB_rate := (1 / 3) / 10

-- Define the combined rate
def combined_rate := PipeA_rate + PipeB_rate

-- Define the time to fill the cistern by both pipes working together
def time_to_fill_cistern := 1 / combined_rate

-- State the theorem to prove
theorem pipes_fill_cistern_together_in_15_minutes :
  time_to_fill_cistern = 15 := by
  sorry

end pipes_fill_cistern_together_in_15_minutes_l260_260621


namespace cyclic_sum_inequality_l260_260724

variable (a b c : ℝ) 
variable (h1 : 4 * a * b * c = a + b + c + 1)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem cyclic_sum_inequality : 
    (a^2 + a) + (b^2 + b) + (c^2 + c) ≥ 2 * (a * b + b * c + c * a) := 
by 
  sorry

end cyclic_sum_inequality_l260_260724


namespace find_y_l260_260539

theorem find_y (y : ℝ) (h : (y^2 - 11 * y + 24) / (y - 1) + (4 * y^2 + 20 * y - 25) / (4*y - 5) = 5) :
  y = 3 ∨ y = 4 :=
sorry

end find_y_l260_260539


namespace calculate_number_l260_260102

theorem calculate_number (tens ones tenths hundredths : ℝ) 
  (h_tens : tens = 21) 
  (h_ones : ones = 8) 
  (h_tenths : tenths = 5) 
  (h_hundredths : hundredths = 34) :
  tens * 10 + ones * 1 + tenths * 0.1 + hundredths * 0.01 = 218.84 :=
by
  sorry

end calculate_number_l260_260102


namespace time_in_1867_minutes_correct_l260_260202

def current_time := (3, 15) -- (hours, minutes)
def minutes_in_hour := 60
def total_minutes := 1867
def hours_after := total_minutes / minutes_in_hour
def remainder_minutes := total_minutes % minutes_in_hour
def result_time := ((current_time.1 + hours_after) % 24, current_time.2 + remainder_minutes)
def expected_time := (22, 22) -- 10:22 p.m. in 24-hour format

theorem time_in_1867_minutes_correct : result_time = expected_time := 
by
    -- No proof is required according to the instructions.
    sorry

end time_in_1867_minutes_correct_l260_260202


namespace correct_statements_l260_260251

-- Definitions from conditions
def probability_of_event (A : Event) : ℝ := sorry -- Assume some function to get the probability

-- Assume some frequency-related function
def stable_frequency_value (A : Event) : ℝ := sorry

-- Assume basic event mutual exclusivity
axiom basic_event_exclusivity (E1 E2 : Event) : (E1 ≠ E2) → (disjoint E1 E2)

-- Axioms for probability bounds
axiom prob_bounds (A : Event) : 0 ≤ probability_of_event A ∧ probability_of_event A ≤ 1

-- The proof statements to be demonstrated
theorem correct_statements (A : Event) :
  (probability_of_event A = stable_frequency_value A) ∧
  (∀ E1 E2, E1 ≠ E2 → disjoint E1 E2) ∧
  (¬ (0 < probability_of_event A ∧ probability_of_event A < 1)) :=
by {
  sorry -- placeholder for proof
}

end correct_statements_l260_260251


namespace rick_ironed_27_pieces_l260_260783

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l260_260783


namespace Alex_runs_faster_l260_260353

def Rick_speed : ℚ := 5
def Jen_speed : ℚ := (3 / 4) * Rick_speed
def Mark_speed : ℚ := (4 / 3) * Jen_speed
def Alex_speed : ℚ := (5 / 6) * Mark_speed

theorem Alex_runs_faster : Alex_speed = 25 / 6 :=
by
  -- Proof is skipped
  sorry

end Alex_runs_faster_l260_260353


namespace find_y_l260_260010

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l260_260010


namespace roots_equality_l260_260768

noncomputable def problem_statement (α β γ δ p q : ℝ) : Prop :=
(α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2 * p - 3 * q) ^ 2

theorem roots_equality (α β γ δ p q : ℝ)
  (h₁ : ∀ x, x^2 - 2 * p * x + 3 = 0 → (x = α ∨ x = β))
  (h₂ : ∀ x, x^2 - 3 * q * x + 4 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
sorry

end roots_equality_l260_260768


namespace transformed_ellipse_l260_260461

-- Define the original equation and the transformation
def orig_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def trans_x (x' : ℝ) : ℝ := x' / 5
noncomputable def trans_y (y' : ℝ) : ℝ := y' / 4

-- Prove that the transformed equation is an ellipse with specified properties
theorem transformed_ellipse :
  (∃ x' y' : ℝ, (trans_x x')^2 + (trans_y y')^2 = 1) →
  ∃ a b : ℝ, (a = 10) ∧ (b = 8) ∧ (∀ x' y' : ℝ, x'^2 / (a/2)^2 + y'^2 / (b/2)^2 = 1) :=
sorry

end transformed_ellipse_l260_260461


namespace sequence_recurrence_l260_260558

theorem sequence_recurrence (a : ℕ → ℝ) (h₀ : a 1 = 1) (h : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  intro n hn
  exact sorry

end sequence_recurrence_l260_260558


namespace parabola_vertex_l260_260486

theorem parabola_vertex {a b c : ℝ} (h₁ : ∃ b c, ∀ x, a * x^2 + b * x + c = a * (x + 3)^2) (h₂ : a * (2 + 3)^2 = -50) : a = -2 :=
by
  sorry

end parabola_vertex_l260_260486


namespace range_of_a_same_solution_set_l260_260369

-- Define the inequality (x-2)(x-5) ≤ 0
def ineq1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the first inequality in the system (x-2)(x-5) ≤ 0
def ineq_system_1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the second inequality in the system x(x-a) ≥ 0
def ineq_system_2 (x a : ℝ) : Prop :=
  x * (x - a) ≥ 0

-- The final proof statement
theorem range_of_a_same_solution_set (a : ℝ) :
  (∀ x : ℝ, ineq_system_1 x ↔ ineq1 x) →
  (∀ x : ℝ, ineq_system_2 x a → ineq1 x) →
  a ≤ 2 :=
sorry

end range_of_a_same_solution_set_l260_260369


namespace winner_lifted_weight_l260_260810

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l260_260810


namespace distance_AC_probability_l260_260394

open Real

theorem distance_AC_probability :
  let A := (0, -10 : ℝ)
  let B := (0, 0 : ℝ)
  let O := (0, -4 : ℝ)
  let C_x (β : ℝ) := 7 * sin β
  let C_y (β : ℝ) := 7 * cos β
  let AC (β : ℝ) := sqrt (C_x β ^ 2 + (C_y β + 10) ^ 2)
  let AO := 6
  ∀ β ∈ Set.Ioo 0 (π/2),
  (AC β < 2 * AO) ↔ β < arc tan(sqrt 371 / 8) := by
  sorry

end distance_AC_probability_l260_260394


namespace special_divisors_count_of_20_30_l260_260161

def prime_number (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def number_of_divisors (a : ℕ) (α β : ℕ) : ℕ := (α + 1) * (β + 1)

def count_special_divisors (m n : ℕ) : ℕ :=
  let total_divisors_m := (m + 1) * (n + 1)
  let total_divisors_n := (n + 1) * (n / 2 + 1)
  (total_divisors_m - 1) / 2 - total_divisors_n + 1

theorem special_divisors_count_of_20_30 (d_20_30 d_20_15 : ℕ) :
  let α := 60
  let β := 30
  let γ := 30
  let δ := 15
  prime_number 2 ∧ prime_number 5 ∧
  count_special_divisors α β = 1891 ∧
  count_special_divisors γ δ = 496 →
  d_20_30 = 2 * 1891 / 2 ∧
  d_20_15 = 2 * 496 →
  count_special_divisors 60 30 - count_special_divisors 30 15 + 1 = 450
:= by
  sorry

end special_divisors_count_of_20_30_l260_260161


namespace solve_equation1_solve_equation2_l260_260066

def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

theorem solve_equation1 :
  (equation1 (-1) ∨ equation1 (1 / 3)) ∧ 
  (∀ x, equation1 x → x = -1 ∨ x = 1 / 3) :=
sorry

theorem solve_equation2 :
  (equation2 1 ∨ equation2 (-4)) ∧ 
  (∀ x, equation2 x → x = 1 ∨ x = -4) :=
sorry

end solve_equation1_solve_equation2_l260_260066


namespace integer_solutions_to_equation_l260_260407

theorem integer_solutions_to_equation :
  ∀ (a b c : ℤ), a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integer_solutions_to_equation_l260_260407


namespace smallest_n_l260_260655

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260655


namespace probability_point_A_on_hyperbola_l260_260060

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l260_260060


namespace area_BNM_l260_260922

-- Define the conditions as hypotheses
def parallelogram (A B C D : Point) : Prop :=
  parallel (line A B) (line C D) ∧ parallel (line B C) (line D A) 

def angle_bisector (A B C M : Point) : Prop :=
  angle ∠BAC = 2 * angle ∠BAM

-- Create the final Lean theorem statement
theorem area_BNM (A B C D M N : Point)
  (h1 : parallelogram A B C D)
  (h2 : distance A B = 6)
  (h3 : height (point D) (line A B) = 3)
  (h4 : on_line A B M)
  (h5 : on_line B C M)
  (h6 : distance B M = 4)
  (h7 : angle_bisector A B B M)
  (h8 : intersection (line A M) (line B D) = N) :
  area (triangle B N M) = 27 / 8 := by
  sorry

end area_BNM_l260_260922


namespace function_continuous_at_x0_l260_260775

noncomputable def delta (ε : ℝ) : ℝ := ε / 36

theorem function_continuous_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 6| < δ → |(3 * x^2 + 7) - 115| < ε :=
by
  -- The following proof list is provided for context and will be replaced by the actual proof using Lean commands.
  -- sorry will be used to indicate the proof is omitted.
  exact sorry

end function_continuous_at_x0_l260_260775


namespace cookie_percentage_increase_l260_260690

theorem cookie_percentage_increase (cookies_Monday cookies_Tuesday cookies_Wednesday total_cookies : ℕ) 
  (h1 : cookies_Monday = 5)
  (h2 : cookies_Tuesday = 2 * cookies_Monday)
  (h3 : total_cookies = cookies_Monday + cookies_Tuesday + cookies_Wednesday)
  (h4 : total_cookies = 29) :
  (100 * (cookies_Wednesday - cookies_Tuesday) / cookies_Tuesday = 40) := 
by
  sorry

end cookie_percentage_increase_l260_260690


namespace largest_4_digit_divisible_by_88_and_prime_gt_100_l260_260817

noncomputable def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem largest_4_digit_divisible_by_88_and_prime_gt_100 (p : ℕ) (hp : is_prime p) (h1 : 100 < p):
  ∃ n, is_4_digit n ∧ is_divisible_by n 88 ∧ is_divisible_by n p ∧
       (∀ m, is_4_digit m ∧ is_divisible_by m 88 ∧ is_divisible_by m p → m ≤ n) :=
sorry

end largest_4_digit_divisible_by_88_and_prime_gt_100_l260_260817


namespace complement_of_union_l260_260559

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 2) * (x + 1) ≤ 0 }
def B : Set ℝ := { x | 0 ≤ x ∧ x < 3 }

theorem complement_of_union :
  Set.compl (A ∪ B) = { x : ℝ | x < -1 } ∪ { x | x ≥ 3 } := by
  sorry

end complement_of_union_l260_260559


namespace three_digit_ends_with_itself_iff_l260_260874

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l260_260874


namespace part1_part2_l260_260561

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem part1 : A ∩ B = {3, 5} := by
  sorry

theorem part2 : (U \ A) ∪ B = {3, 4, 5, 6} := by
  sorry

end part1_part2_l260_260561


namespace retail_store_paid_40_percent_more_l260_260675

variables (C R : ℝ)

-- Condition: The customer price is 96% more than manufacturing cost
def customer_price_from_manufacturing (C : ℝ) : ℝ := 1.96 * C

-- Condition: The customer price is 40% more than the retailer price
def customer_price_from_retail (R : ℝ) : ℝ := 1.40 * R

-- Theorem to be proved
theorem retail_store_paid_40_percent_more (C R : ℝ) 
  (h_customer_price : customer_price_from_manufacturing C = customer_price_from_retail R) :
  (R - C) / C = 0.40 :=
by
  sorry

end retail_store_paid_40_percent_more_l260_260675


namespace Mart_income_percentage_of_Juan_l260_260340

variable (J T M : ℝ)

-- Conditions
def Tim_income_def : Prop := T = 0.5 * J
def Mart_income_def : Prop := M = 1.6 * T

-- Theorem to prove
theorem Mart_income_percentage_of_Juan
  (h1 : Tim_income_def T J) 
  (h2 : Mart_income_def M T) : 
  (M / J) * 100 = 80 :=
by
  sorry

end Mart_income_percentage_of_Juan_l260_260340


namespace arcsin_one_half_eq_pi_six_l260_260121

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end arcsin_one_half_eq_pi_six_l260_260121


namespace kite_height_30_sqrt_43_l260_260882

theorem kite_height_30_sqrt_43
  (c d h : ℝ)
  (h1 : h^2 + c^2 = 170^2)
  (h2 : h^2 + d^2 = 150^2)
  (h3 : c^2 + d^2 = 160^2) :
  h = 30 * Real.sqrt 43 := by
  sorry

end kite_height_30_sqrt_43_l260_260882


namespace rhombus_area_of_square_4_l260_260758

theorem rhombus_area_of_square_4 :
  let A := (0, 4)
  let B := (0, 0)
  let C := (4, 0)
  let D := (4, 4)
  let F := (0, 2)  -- Midpoint of AB
  let E := (4, 2)  -- Midpoint of CD
  let FG := 2 -- Half of the side of the square (since F and E are midpoints)
  let GH := 2
  let HE := 2
  let EF := 2
  let rhombus_FGEH_area := 1 / 2 * FG * EH
  rhombus_FGEH_area = 4 := sorry

end rhombus_area_of_square_4_l260_260758


namespace circle_x_intersect_l260_260071

theorem circle_x_intersect (x y : ℝ) : 
  (x, y) = (0, 0) ∨ (x, y) = (10, 0) → (x = 10) :=
by
  -- conditions:
  -- The endpoints of the diameter are (0,0) and (10,10)
  -- (proving that the second intersect on x-axis has x-coordinate 10)
  sorry

end circle_x_intersect_l260_260071


namespace ratio_of_areas_of_shaded_and_white_region_l260_260972

theorem ratio_of_areas_of_shaded_and_white_region
  (all_squares_have_vertices_in_middle: ∀ (n : ℕ), n ≠ 0 → (square_vertices_positioned_mid : Prop)) :
  ∃ (ratio : ℚ), ratio = 5 / 3 :=
by
  sorry

end ratio_of_areas_of_shaded_and_white_region_l260_260972


namespace cantor_length_formula_l260_260260

noncomputable def cantor_length : ℕ → ℚ
| 0 => 1
| (n+1) => 2/3 * cantor_length n

theorem cantor_length_formula (n : ℕ) : cantor_length n = (2/3 : ℚ)^(n-1) :=
  sorry

end cantor_length_formula_l260_260260


namespace balls_in_boxes_l260_260570

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l260_260570


namespace total_clothing_ironed_l260_260785

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l260_260785


namespace jenny_proposal_time_l260_260171

theorem jenny_proposal_time (total_time research_time report_time proposal_time : ℕ) 
  (h1 : total_time = 20) 
  (h2 : research_time = 10) 
  (h3 : report_time = 8) 
  (h4 : proposal_time = total_time - research_time - report_time) : 
  proposal_time = 2 := 
by
  sorry

end jenny_proposal_time_l260_260171


namespace rabbit_carrots_l260_260753

theorem rabbit_carrots (h_r h_f x : ℕ) (H1 : 5 * h_r = x) (H2 : 6 * h_f = x) (H3 : h_r = h_f + 2) : x = 60 :=
by
  sorry

end rabbit_carrots_l260_260753


namespace smallest_n_45_l260_260631

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260631


namespace selling_price_for_loss_l260_260207

noncomputable def cp : ℝ := 640
def sp1 : ℝ := 768
def sp2 : ℝ := 448
def sp_profitable_sale : ℝ := 832

theorem selling_price_for_loss :
  sp_profitable_sale - cp = cp - sp2 :=
by
  sorry

end selling_price_for_loss_l260_260207


namespace bruce_goals_l260_260823

theorem bruce_goals (B M : ℕ) (h1 : M = 3 * B) (h2 : B + M = 16) : B = 4 :=
by {
  -- Omitted proof
  sorry
}

end bruce_goals_l260_260823


namespace A_inter_B_l260_260328

-- Define the sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := { abs x | x ∈ A }

-- Statement of the theorem to be proven
theorem A_inter_B :
  A ∩ B = {0, 2} := 
by 
  sorry

end A_inter_B_l260_260328


namespace painter_red_cells_count_l260_260620

open Nat

/-- Prove the number of red cells painted by the painter in the given 2000 x 70 grid. -/
theorem painter_red_cells_count :
  let rows := 2000
  let columns := 70
  let lcm_rc := Nat.lcm rows columns -- Calculate the LCM of row and column counts
  lcm_rc = 14000 := by
sorry

end painter_red_cells_count_l260_260620


namespace nancy_picked_l260_260250

variable (total_picked : ℕ) (alyssa_picked : ℕ)

-- Assuming the conditions given in the problem
def conditions := total_picked = 59 ∧ alyssa_picked = 42

-- Proving that Nancy picked 17 pears
theorem nancy_picked : conditions total_picked alyssa_picked → total_picked - alyssa_picked = 17 := by
  sorry

end nancy_picked_l260_260250


namespace largest_integer_value_l260_260879

theorem largest_integer_value (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 2/3 ∧ (x : ℚ) < 10 → x = 3 := 
by
  sorry

end largest_integer_value_l260_260879


namespace minimal_sum_of_squares_l260_260467

theorem minimal_sum_of_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ p q r : ℕ, a + b = p^2 ∧ b + c = q^2 ∧ a + c = r^2) ∧
  a + b + c = 55 := 
by sorry

end minimal_sum_of_squares_l260_260467


namespace gum_distribution_l260_260035

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l260_260035


namespace max_product_distance_l260_260429

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def is_focus (F : ℝ × ℝ) : Prop := F = (3, 0) ∨ F = (-3, 0)

-- The theorem statement
theorem max_product_distance (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : ellipse M.1 M.2) 
  (h2 : is_focus F1) 
  (h3 : is_focus F2) : 
  (∃ x y, M = (x, y) ∧ ellipse x y) → 
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2| * |(M.1 - F2.1)^2 + (M.2 - F2.2)^2| ≤ 81 := 
sorry

end max_product_distance_l260_260429


namespace solve_factorial_equation_in_natural_numbers_l260_260065

theorem solve_factorial_equation_in_natural_numbers :
  ∃ n k : ℕ, n! + 3 * n + 8 = k^2 ↔ n = 2 ∧ k = 4 := by
sorry

end solve_factorial_equation_in_natural_numbers_l260_260065


namespace negation_prop_l260_260075

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l260_260075


namespace triple_hash_90_l260_260534

def hash (N : ℝ) : ℝ := 0.3 * N + 2

theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 :=
by
  sorry

end triple_hash_90_l260_260534


namespace mrs_jane_total_coins_l260_260111

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l260_260111


namespace find_smallest_n_l260_260635

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260635


namespace kopecks_payment_l260_260476

theorem kopecks_payment (n : ℕ) (h : n ≥ 8) : ∃ (a b : ℕ), n = 3 * a + 5 * b :=
sorry

end kopecks_payment_l260_260476


namespace sum_is_eight_l260_260603

theorem sum_is_eight (a b c d : ℤ)
  (h1 : 2 * (a - b + c) = 10)
  (h2 : 2 * (b - c + d) = 12)
  (h3 : 2 * (c - d + a) = 6)
  (h4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 :=
by
  sorry

end sum_is_eight_l260_260603


namespace value_of_expression_l260_260438

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : x^2 + 3*x - 2 = 0 := 
by {
  -- proof logic will be here
  sorry
}

end value_of_expression_l260_260438


namespace solve_for_n_l260_260264

variable (n : ℚ)

theorem solve_for_n (h : 22 + Real.sqrt (-4 + 18 * n) = 24) : n = 4 / 9 := by
  sorry

end solve_for_n_l260_260264


namespace simplify_expression_correct_l260_260355

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + Real.sqrt 5) - 2 / (2 - Real.sqrt 5)

theorem simplify_expression_correct : simplify_expression = 10 := by
  sorry

end simplify_expression_correct_l260_260355


namespace perfect_square_2n_plus_65_l260_260270

theorem perfect_square_2n_plus_65 (n : ℕ) (h : n > 0) : 
  (∃ m : ℕ, m * m = 2^n + 65) → n = 4 ∨ n = 10 :=
by 
  sorry

end perfect_square_2n_plus_65_l260_260270


namespace internet_plan_cost_effective_l260_260479

theorem internet_plan_cost_effective (d : ℕ) :
  (∀ (d : ℕ), d > 150 → 1500 + 10 * d < 20 * d) ↔ d = 151 :=
sorry

end internet_plan_cost_effective_l260_260479


namespace combined_swim_time_l260_260259

theorem combined_swim_time 
    (freestyle_time: ℕ)
    (backstroke_without_factors: ℕ)
    (backstroke_with_factors: ℕ)
    (butterfly_without_factors: ℕ)
    (butterfly_with_factors: ℕ)
    (breaststroke_without_factors: ℕ)
    (breaststroke_with_factors: ℕ) :
    freestyle_time = 48 ∧
    backstroke_without_factors = freestyle_time + 4 ∧
    backstroke_with_factors = backstroke_without_factors + 2 ∧
    butterfly_without_factors = backstroke_without_factors + 3 ∧
    butterfly_with_factors = butterfly_without_factors + 3 ∧
    breaststroke_without_factors = butterfly_without_factors + 2 ∧
    breaststroke_with_factors = breaststroke_without_factors - 1 →
    freestyle_time + backstroke_with_factors + butterfly_with_factors + breaststroke_with_factors = 216 :=
by
  sorry

end combined_swim_time_l260_260259


namespace mike_spent_total_l260_260345

-- Define the prices of the items
def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84

-- Define the total price calculation
def total_price : ℝ := trumpet_price + song_book_price

-- The theorem statement asserting the total price
theorem mike_spent_total : total_price = 151.00 :=
by
  sorry

end mike_spent_total_l260_260345


namespace part_1_part_2_part_3_l260_260156

def f (x : ℝ) : ℝ := sin x ^ 2 - cos x ^ 2 - 2 * sqrt 3 * sin x * cos x

theorem part_1 : f (2 * π / 3) = 2 := 
sorry

theorem part_2 : ∀ x : ℝ, f (x + π) = f x := 
sorry

theorem part_3 : ∀ k : ℤ, ∀ x : ℝ, 
  k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 3 → 
  2 * sin (2 * x + 7 * π / 6) < 2 * sin (2 * (x + 1) + 7 * π / 6) := 
sorry

end part_1_part_2_part_3_l260_260156


namespace cheaper_store_price_in_cents_l260_260439

/-- List price of Book Y -/
def list_price : ℝ := 24.95

/-- Discount at Readers' Delight -/
def readers_delight_discount : ℝ := 5

/-- Discount rate at Book Bargains -/
def book_bargains_discount_rate : ℝ := 0.2

/-- Calculate sale price at Readers' Delight -/
def sale_price_readers_delight : ℝ := list_price - readers_delight_discount

/-- Calculate sale price at Book Bargains -/
def sale_price_book_bargains : ℝ := list_price * (1 - book_bargains_discount_rate)

/-- Difference in price between Book Bargains and Readers' Delight in cents -/
theorem cheaper_store_price_in_cents :
  (sale_price_book_bargains - sale_price_readers_delight) * 100 = 1 :=
by
  sorry

end cheaper_store_price_in_cents_l260_260439


namespace area_of_rhombus_l260_260357

noncomputable def diagonal_length_1 : ℕ := 30
noncomputable def diagonal_length_2 : ℕ := 14

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = diagonal_length_1) (h2 : d2 = diagonal_length_2) : 
  (d1 * d2) / 2 = 210 :=
by 
  rw [h1, h2]
  sorry

end area_of_rhombus_l260_260357


namespace add_2001_1015_l260_260233

theorem add_2001_1015 : 2001 + 1015 = 3016 := 
by
  sorry

end add_2001_1015_l260_260233


namespace find_weight_of_a_l260_260385

variables (a b c d e : ℕ)

-- Conditions
def cond1 : Prop := a + b + c = 252
def cond2 : Prop := a + b + c + d = 320
def cond3 : Prop := e = d + 7
def cond4 : Prop := b + c + d + e = 316

theorem find_weight_of_a (h1 : cond1 a b c) (h2 : cond2 a b c d) (h3 : cond3 d e) (h4 : cond4 b c d e) :
  a = 79 :=
by sorry

end find_weight_of_a_l260_260385


namespace arcsin_one_half_l260_260122

theorem arcsin_one_half : real.arcsin (1 / 2) = real.pi / 6 :=
by
  sorry

end arcsin_one_half_l260_260122


namespace fraction_sum_l260_260232

theorem fraction_sum :
  (1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 : ℚ) = -5 / 6 :=
by sorry

end fraction_sum_l260_260232


namespace pictures_left_l260_260381

def zoo_pics : ℕ := 802
def museum_pics : ℕ := 526
def beach_pics : ℕ := 391
def amusement_park_pics : ℕ := 868
def duplicates_deleted : ℕ := 1395

theorem pictures_left : 
  (zoo_pics + museum_pics + beach_pics + amusement_park_pics - duplicates_deleted) = 1192 := 
by
  sorry

end pictures_left_l260_260381


namespace binomial_expectation_variance_l260_260278

noncomputable theory

open ProbabilityTheory

variables {n : ℕ} {p : ℚ} 

theorem binomial_expectation_variance 
  (X : ℕ → ℚ) (hn : X ∼ binomial n p) 
  (hE : 3 * (E[X] : ℚ) - 9 = 27)
  (hD : 9 * (Var[X] : ℚ) = 27) :
  n = 16 ∧ p = 3 / 4 := 
sorry

end binomial_expectation_variance_l260_260278


namespace sum_of_two_squares_l260_260774

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 :=
by sorry

end sum_of_two_squares_l260_260774


namespace A_cubed_inv_l260_260740

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Given condition
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 7], ![-2, -4]]

-- Goal to prove
theorem A_cubed_inv :
  (A^3)⁻¹ = ![![11, 17], ![2, 6]] :=
  sorry

end A_cubed_inv_l260_260740


namespace roots_sum_of_squares_of_pairs_l260_260046

noncomputable def roots (p q r : ℝ) : Prop := 
  ∃ (p q r : ℝ), (p, q, r ∈ Roots (x^3 - 15*x^2 + 25*x - 10))

theorem roots_sum_of_squares_of_pairs (p q r : ℝ) (h : roots p q r) : 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by
  sorry

end roots_sum_of_squares_of_pairs_l260_260046


namespace apple_distribution_l260_260113

theorem apple_distribution : 
  ∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 2 ∧ 
  ∃ n : ℕ, n = nat.choose 25 2 ∧ n = 300 :=
by
  use 3
  use 2
  use 2
  have : 3 + 2 + 2 ≤ 30 := by norm_num
  have : nat.choose 25 2 = 300 := by
    rw [nat.choose_eq_factorial_div_factorial 24 1 2]
    norm_num
  use 300
  sorry

end apple_distribution_l260_260113


namespace algebraic_expression_value_l260_260140

theorem algebraic_expression_value (b a c : ℝ) (h₁ : b < a) (h₂ : a < 0) (h₃ : 0 < c) :
  |b| - |b - a| + |c - a| - |a + b| = b + c - a :=
by
  sorry

end algebraic_expression_value_l260_260140


namespace percentage_increase_l260_260530

variables (P : ℝ) (buy_price : ℝ := 0.60 * P) (sell_price : ℝ := 1.08000000000000007 * P)

theorem percentage_increase (h: (0.60 : ℝ) * P = buy_price) (h1: (1.08000000000000007 : ℝ) * P = sell_price) :
  ((sell_price - buy_price) / buy_price) * 100 = 80.00000000000001 :=
  sorry

end percentage_increase_l260_260530


namespace frosting_time_difference_l260_260937

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l260_260937


namespace at_least_one_not_less_than_four_l260_260841

theorem at_least_one_not_less_than_four 
( m n t : ℝ ) 
( h_m : 0 < m ) 
( h_n : 0 < n ) 
( h_t : 0 < t ) : 
∃ a, ( a = m + 4 / n ∨ a = n + 4 / t ∨ a = t + 4 / m ) ∧ 4 ≤ a :=
sorry

end at_least_one_not_less_than_four_l260_260841


namespace seating_arrangements_exactly_two_adjacent_empty_l260_260372

theorem seating_arrangements_exactly_two_adjacent_empty :
  let seats := 6
  let people := 3
  let arrangements := (seats.factorial / (seats - people).factorial)
  let non_adj_non_empty := ((seats - people).choose people * people.factorial)
  let all_adj_empty := ((seats - (people + 1)).choose 1 * people.factorial)
  arrangements - non_adj_non_empty - all_adj_empty = 72 := by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_l260_260372


namespace ron_total_tax_l260_260257

def car_price : ℝ := 30000
def first_tier_level : ℝ := 10000
def first_tier_rate : ℝ := 0.25
def second_tier_rate : ℝ := 0.15

def first_tier_tax : ℝ := first_tier_level * first_tier_rate
def second_tier_tax : ℝ := (car_price - first_tier_level) * second_tier_rate
def total_tax : ℝ := first_tier_tax + second_tier_tax

theorem ron_total_tax : 
  total_tax = 5500 := by
  -- Proof will be provided here
  sorry

end ron_total_tax_l260_260257


namespace polar_to_cartesian_parabola_l260_260126

theorem polar_to_cartesian_parabola (r θ : ℝ) (h : r = 1 / (1 - Real.sin θ)) :
  ∃ x y : ℝ, x^2 = 2 * y + 1 :=
by
  sorry

end polar_to_cartesian_parabola_l260_260126


namespace ratio_of_supply_to_demand_l260_260024

theorem ratio_of_supply_to_demand (supply demand : ℕ)
  (hs : supply = 1800000)
  (hd : demand = 2400000) :
  supply / (Nat.gcd supply demand) = 3 ∧ demand / (Nat.gcd supply demand) = 4 :=
by
  sorry

end ratio_of_supply_to_demand_l260_260024


namespace sum_of_distinct_prime_factors_l260_260712

theorem sum_of_distinct_prime_factors (a b c : ℕ) (h1 : a = 7^4 - 7^2) (h2 : b = 2) (h3 : c = 3) (h4 : 2 + 3 + 7 = 12): 
  ∃ d : ℕ, a.prime_factors.sum = d ∧ d = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l260_260712


namespace selection_with_boys_and_girls_l260_260717

def boys := 4
def girls := 3
def total := boys + girls
def choose_total := Nat.choose total 4
def choose_boys_only := Nat.choose boys 4

theorem selection_with_boys_and_girls :
  choose_total - choose_boys_only = 34 :=
by
  -- Proof goes here
  sorry

end selection_with_boys_and_girls_l260_260717


namespace binary_101110_to_octal_l260_260123

-- Definition: binary number 101110 represents some decimal number
def binary_101110 : ℕ := 0 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5

-- Definition: decimal number 46 represents some octal number
def decimal_46 := 46

-- A utility function to convert decimal to octal (returns the digits as a list)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else decimal_to_octal (n / 8) ++ [n % 8]

-- Hypothesis: the binary 101110 equals the decimal 46
lemma binary_101110_eq_46 : binary_101110 = decimal_46 := by sorry

-- Hypothesis: the decimal 46 converts to the octal number 56 (in list form)
def octal_56 := [5, 6]

-- Theorem: binary 101110 converts to the octal number 56
theorem binary_101110_to_octal :
  decimal_to_octal binary_101110 = octal_56 := by
  rw [binary_101110_eq_46]
  sorry

end binary_101110_to_octal_l260_260123


namespace marks_for_correct_answer_l260_260577

theorem marks_for_correct_answer (x : ℕ) 
  (total_marks : ℤ) (total_questions : ℕ) (correct_answers : ℕ) 
  (wrong_mark : ℤ) (result : ℤ) :
  total_marks = result →
  total_questions = 70 →
  correct_answers = 27 →
  (-1) * (total_questions - correct_answers) = wrong_mark →
  total_marks = (correct_answers : ℤ) * (x : ℤ) + wrong_mark →
  x = 3 := 
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end marks_for_correct_answer_l260_260577


namespace intersection_of_sets_l260_260736

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def set_B : Set ℝ := {x | (x + 1) * (x - 4) > 0}

theorem intersection_of_sets :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ {x | (x + 1) * (x - 4) > 0} = {x | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l260_260736


namespace find_vector_l260_260699

def line_r (t : ℝ) : ℝ × ℝ :=
  (2 + 5 * t, 3 - 2 * t)

def line_s (u : ℝ) : ℝ × ℝ :=
  (1 + 5 * u, -2 - 2 * u)

def is_projection (w1 w2 : ℝ) : Prop :=
  w1 - w2 = 3

theorem find_vector (w1 w2 : ℝ) (h_proj : is_projection w1 w2) :
  (w1, w2) = (-2, -5) :=
sorry

end find_vector_l260_260699


namespace number_of_ways_to_put_7_balls_in_2_boxes_l260_260563

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l260_260563


namespace probability_of_two_draws_l260_260521

theorem probability_of_two_draws (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
    (total_balls = white_balls + black_balls) 
    (P_black_first : ℚ := black_balls / total_balls) 
    (remaining_balls_after_black : ℕ := total_balls - 1) 
    (P_white_second := white_balls / remaining_balls_after_black) :
     P_black_first * P_white_second = 7 / 30 :=
by
  have total_eq : total_balls = 7 + 3 := by simp [white_balls, black_balls]
  have prob_black_first : P_black_first = 3 / 10 := by simp [P_black_first, total_eq]
  have remaining_balls_eq : remaining_balls_after_black = 9 := by simp [remaining_balls_after_black, total_eq]
  have prob_white_second : P_white_second == 7 / remaining_balls_after_black := by simp [P_white_second, total_eq]
  have prob_white_second_final := 7 / 9 := by simp [prob_white_second, remaining_balls_eq]
  sorry

end probability_of_two_draws_l260_260521


namespace volume_tetrahedron_lt_one_eighth_l260_260206

noncomputable def volume_of_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * |((B.1 - A.1) * ((C.2 - A.2) * (D.3 - A.3) - (C.3 - A.3) * (D.2 - A.2)) -
             (B.2 - A.2) * ((C.1 - A.1) * (D.3 - A.3) - (C.3 - A.3) * (D.1 - A.1)) +
             (B.3 - A.3) * ((C.1 - A.1) * (D.2 - A.2) - (C.2 - A.2) * (D.1 - A.1)))|

theorem volume_tetrahedron_lt_one_eighth
  (A B C D : ℝ × ℝ × ℝ)
  (hAB : dist A B < 1)
  (hAC : dist A C < 1)
  (hAD : dist A D < 1)
  (hBC : dist B C < 1)
  (hBD : dist B D < 1) :
  volume_of_tetrahedron A B C D < 1 / 8 :=
by
  sorry

end volume_tetrahedron_lt_one_eighth_l260_260206


namespace smallest_n_satisfies_conditions_l260_260647

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260647


namespace serena_mother_age_l260_260600

theorem serena_mother_age {x : ℕ} (h : 39 + x = 3 * (9 + x)) : x = 6 := 
by
  sorry

end serena_mother_age_l260_260600


namespace isosceles_right_triangle_leg_length_l260_260613

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l260_260613


namespace average_of_four_l260_260302

variable {r s t u : ℝ}

theorem average_of_four (h : (5 / 2) * (r + s + t + u) = 20) : (r + s + t + u) / 4 = 2 := 
by 
  sorry

end average_of_four_l260_260302


namespace amount_spent_on_first_shop_l260_260053

-- Define the conditions
def booksFromFirstShop : ℕ := 65
def costFromSecondShop : ℕ := 2000
def booksFromSecondShop : ℕ := 35
def avgPricePerBook : ℕ := 85

-- Calculate the total books and the total amount spent
def totalBooks : ℕ := booksFromFirstShop + booksFromSecondShop
def totalAmountSpent : ℕ := totalBooks * avgPricePerBook

-- Prove the amount spent on the books from the first shop is Rs. 6500
theorem amount_spent_on_first_shop : 
  (totalAmountSpent - costFromSecondShop) = 6500 :=
by
  sorry

end amount_spent_on_first_shop_l260_260053


namespace max_value_3x_4y_l260_260551

noncomputable def y_geom_mean (x y : ℝ) : Prop :=
  y^2 = (1 - x) * (1 + x)

theorem max_value_3x_4y (x y : ℝ) (h : y_geom_mean x y) : 3 * x + 4 * y ≤ 5 :=
sorry

end max_value_3x_4y_l260_260551


namespace even_function_value_l260_260549

-- Define the function condition
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the main problem with given conditions
theorem even_function_value (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : ∀ x : ℝ, x < 0 → f x = x * (x + 1)) 
  (x : ℝ) (hx : x > 0) : f x = x * (x - 1) :=
  sorry

end even_function_value_l260_260549


namespace correct_student_answer_l260_260455

theorem correct_student_answer :
  (9 - (3^2) / 8 = 9 - (9 / 8)) ∧
  (24 - (4 * (3^2)) = 24 - 36) ∧
  ((36 - 12) / (3 / 2) = 24 * (2 / 3)) ∧
  ((-3)^2 / (1 / 3) * 3 = 9 * 3 * 3) →
  (24 * (2 / 3) = 16) :=
by
  sorry

end correct_student_answer_l260_260455


namespace puppies_start_count_l260_260393

theorem puppies_start_count (x : ℕ) (given_away : ℕ) (left : ℕ) (h1 : given_away = 7) (h2 : left = 5) (h3 : x = given_away + left) : x = 12 :=
by
  rw [h1, h2] at h3
  exact h3

end puppies_start_count_l260_260393


namespace base_rate_of_first_company_is_7_l260_260110

noncomputable def telephone_company_base_rate_proof : Prop :=
  ∃ (base_rate1 base_rate2 charge_per_minute1 charge_per_minute2 minutes : ℝ),
  base_rate1 = 7 ∧
  charge_per_minute1 = 0.25 ∧
  base_rate2 = 12 ∧
  charge_per_minute2 = 0.20 ∧
  minutes = 100 ∧
  (base_rate1 + charge_per_minute1 * minutes) =
  (base_rate2 + charge_per_minute2 * minutes) ∧
  base_rate1 = 7

theorem base_rate_of_first_company_is_7 :
  telephone_company_base_rate_proof :=
by
  -- The proof step will go here
  sorry

end base_rate_of_first_company_is_7_l260_260110


namespace maximum_k_l260_260715

open Finset

def not_power_of_three (n : ℕ) : Prop :=
  ∀ k : ℕ, 3 ^ k ≠ n

theorem maximum_k (S : Finset ℕ) (hS : S = (range 243).erase 0) :
  ∃ T ⊆ S, T.card = 121 ∧ ∀ a b ∈ T, not_power_of_three (a + b) := sorry

end maximum_k_l260_260715


namespace problem_statement_l260_260725

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - 2 / x^2 + a / x

theorem problem_statement (a : ℝ) (k : ℝ) : 
  0 < a ∧ a ≤ 4 →
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 →
  |f x1 a - f x2 a| > k * |x1 - x2|) ↔
  k ≤ 2 - a^3 / 108 :=
by
  sorry

end problem_statement_l260_260725


namespace tan_alpha_tan_beta_l260_260905

/-- Given the cosine values of the sum and difference of two angles, 
    find the value of the product of their tangents. -/
theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := sorry

end tan_alpha_tan_beta_l260_260905


namespace amount_of_water_formed_l260_260711

-- Define chemical compounds and reactions
def NaOH : Type := Unit
def HClO4 : Type := Unit
def NaClO4 : Type := Unit
def H2O : Type := Unit

-- Define the balanced chemical equation
def balanced_reaction (n_NaOH n_HClO4 : Int) : (n_NaOH = n_HClO4) → (n_NaOH = 1 → n_HClO4 = 1 → Int × Int × Int × Int) :=
  λ h_ratio h_NaOH h_HClO4 => 
    (n_NaOH, n_HClO4, 1, 1)  -- 1 mole of NaOH reacts with 1 mole of HClO4 to form 1 mole of NaClO4 and 1 mole of H2O

noncomputable def molar_mass_H2O : Float := 18.015 -- g/mol

theorem amount_of_water_formed :
  ∀ (n_NaOH n_HClO4 : Int), 
  (n_NaOH = 1 ∧ n_HClO4 = 1) →
  ((n_NaOH = n_HClO4) → molar_mass_H2O = 18.015) :=
by
  intros n_NaOH n_HClO4 h_condition h_ratio
  sorry

end amount_of_water_formed_l260_260711


namespace solve_arithmetic_seq_l260_260601

theorem solve_arithmetic_seq (x : ℝ) (h : x > 0) (hx : x^2 = (4 + 16) / 2) : x = Real.sqrt 10 :=
sorry

end solve_arithmetic_seq_l260_260601


namespace range_of_c_l260_260888

theorem range_of_c (a b c : ℝ) (h1 : 6 < a) (h2 : a < 10) (h3 : a / 2 ≤ b) (h4 : b ≤ 2 * a) (h5 : c = a + b) : 
  9 < c ∧ c < 30 :=
sorry

end range_of_c_l260_260888


namespace minimum_value_w_l260_260509

theorem minimum_value_w : 
  ∀ x y : ℝ, ∃ (w : ℝ), w = 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 → w ≥ 26.25 :=
by
  intro x y
  use 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30
  sorry

end minimum_value_w_l260_260509


namespace express_as_sum_of_cubes_l260_260857

variables {a b : ℝ}

theorem express_as_sum_of_cubes (a b : ℝ) : 
  2 * a * (a^2 + 3 * b^2) = (a + b)^3 + (a - b)^3 :=
by sorry

end express_as_sum_of_cubes_l260_260857


namespace common_number_in_sequences_l260_260685

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end common_number_in_sequences_l260_260685


namespace glove_problem_l260_260618

theorem glove_problem : 
  let total_ways := nat.choose 8 4
  let no_pair_ways := 2 * 2 * 2 * 2
  total_ways - no_pair_ways = 54 :=
by
  sorry

end glove_problem_l260_260618


namespace value_of_a_l260_260968

theorem value_of_a (a b : ℝ) (h1 : b = 2120) (h2 : a / b = 0.5) : a = 1060 := 
by
  sorry

end value_of_a_l260_260968


namespace find_smallest_n_l260_260640

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260640


namespace largest_common_term_in_range_1_to_200_l260_260687

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l260_260687


namespace trigonometric_ineq_l260_260959

theorem trigonometric_ineq (h₁ : (Real.pi / 4) < 1.5) (h₂ : 1.5 < (Real.pi / 2)) : 
  Real.cos 1.5 < Real.sin 1.5 ∧ Real.sin 1.5 < Real.tan 1.5 := 
sorry

end trigonometric_ineq_l260_260959


namespace ping_pong_shaved_head_ping_pong_upset_l260_260460

noncomputable def probability_shaved_head (pA pB : ℚ) : ℚ :=
  pA^3 + pB^3

noncomputable def probability_upset (pB pA : ℚ) : ℚ :=
  (pB^3) + (3 * (pB^2) * pA) + (6 * (pA^2) * (pB^2))

theorem ping_pong_shaved_head :
  probability_shaved_head (2/3) (1/3) = 1/3 := 
by
  sorry

theorem ping_pong_upset :
  probability_upset (1/3) (2/3) = 11/27 := 
by
  sorry

end ping_pong_shaved_head_ping_pong_upset_l260_260460


namespace quadratic_roots_range_l260_260886

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (x1^2 - 2 * x1 + m - 2 = 0) ∧ 
    (x2^2 - 2 * x2 + m - 2 = 0)) → m < 3 := 
by 
  sorry

end quadratic_roots_range_l260_260886


namespace evaluate_power_l260_260413

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l260_260413


namespace sum_base5_eq_l260_260426

theorem sum_base5_eq :
  (432 + 43 + 4 : ℕ) = 1034 :=
by sorry

end sum_base5_eq_l260_260426


namespace A_work_days_l260_260842

theorem A_work_days {total_wages B_share : ℝ} (B_work_days : ℝ) (total_wages_eq : total_wages = 5000) 
    (B_share_eq : B_share = 3333) (B_rate : ℝ) (correct_rate : B_rate = 1 / B_work_days) :
    ∃x : ℝ, B_share / (total_wages - B_share) = B_rate / (1 / x) ∧ total_wages - B_share = 5000 - B_share ∧ B_work_days = 10 -> x = 20 :=
by
  sorry

end A_work_days_l260_260842


namespace brian_holds_breath_for_60_seconds_l260_260395

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end brian_holds_breath_for_60_seconds_l260_260395


namespace min_f_value_l260_260472

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2

theorem min_f_value (x y z : ℝ) (hxyz_pos : 0 < x ∧ 0 < y ∧ 0 < z) (hxyz : x * y * z = 1) :
  f x y z ≥ 18 :=
sorry

end min_f_value_l260_260472


namespace necessary_condition_for_A_l260_260293

variable {x a : ℝ}

def A : Set ℝ := { x | (x - 2) / (x + 1) ≤ 0 }

theorem necessary_condition_for_A (x : ℝ) (h : x ∈ A) (ha : x ≥ a) : a ≤ -1 :=
sorry

end necessary_condition_for_A_l260_260293


namespace area_square_EFGH_equiv_144_l260_260691

theorem area_square_EFGH_equiv_144 (a b : ℝ) (h : a = 6) (hb : b = 6)
  (side_length_EFGH : ℝ) (hs : side_length_EFGH = a + 3 + 3) : side_length_EFGH ^ 2 = 144 :=
by
  -- Given conditions
  sorry

end area_square_EFGH_equiv_144_l260_260691


namespace jake_peaches_is_7_l260_260067

variable (Steven_peaches Jake_peaches Jill_peaches : ℕ)

-- Conditions:
def Steven_has_19_peaches : Steven_peaches = 19 := by sorry

def Jake_has_12_fewer_peaches_than_Steven : Jake_peaches = Steven_peaches - 12 := by sorry

def Jake_has_72_more_peaches_than_Jill : Jake_peaches = Jill_peaches + 72 := by sorry

-- Proof problem:
theorem jake_peaches_is_7 
    (Steven_peaches Jake_peaches Jill_peaches : ℕ)
    (h1 : Steven_peaches = 19)
    (h2 : Jake_peaches = Steven_peaches - 12)
    (h3 : Jake_peaches = Jill_peaches + 72) :
    Jake_peaches = 7 := by sorry

end jake_peaches_is_7_l260_260067


namespace triangle_side_length_l260_260923

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h₁ : a * Real.cos B = b * Real.sin A)
  (h₂ : C = Real.pi / 6) (h₃ : c = 2) : b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l260_260923


namespace equilateral_triangle_data_l260_260606

theorem equilateral_triangle_data
  (A : ℝ)
  (b : ℝ)
  (ha : A = 450)
  (hb : b = 25)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a) :
  ∃ (h P : ℝ), h = 36 ∧ P = 75 := by
  sorry

end equilateral_triangle_data_l260_260606


namespace find_positive_X_l260_260041

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l260_260041


namespace evaluate_power_l260_260414

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l260_260414


namespace find_smallest_n_l260_260638

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260638


namespace remainder_when_divided_by_x_minus_2_l260_260510

noncomputable def f (x : ℝ) : ℝ :=
  x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 18

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 22 := 
sorry

end remainder_when_divided_by_x_minus_2_l260_260510


namespace remaining_credit_to_be_paid_l260_260589

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l260_260589


namespace average_cost_per_trip_is_correct_l260_260773

def oldest_pass_cost : ℕ := 100
def second_oldest_pass_cost : ℕ := 90
def third_oldest_pass_cost : ℕ := 80
def youngest_pass_cost : ℕ := 70

def oldest_trips : ℕ := 35
def second_oldest_trips : ℕ := 25
def third_oldest_trips : ℕ := 20
def youngest_trips : ℕ := 15

def total_cost : ℕ := oldest_pass_cost + second_oldest_pass_cost + third_oldest_pass_cost + youngest_pass_cost
def total_trips : ℕ := oldest_trips + second_oldest_trips + third_oldest_trips + youngest_trips

def average_cost_per_trip : ℚ := total_cost / total_trips

theorem average_cost_per_trip_is_correct : average_cost_per_trip = 340 / 95 :=
by sorry

end average_cost_per_trip_is_correct_l260_260773


namespace probability_of_darkness_l260_260832

theorem probability_of_darkness (rev_per_min : ℕ) (stay_in_dark_time : ℕ) (revolution_time : ℕ) (stay_fraction : ℕ → ℚ) :
  rev_per_min = 2 →
  stay_in_dark_time = 10 →
  revolution_time = 60 / rev_per_min →
  stay_fraction stay_in_dark_time / revolution_time = 1 / 3 :=
by
  sorry

end probability_of_darkness_l260_260832


namespace junk_mail_per_block_l260_260214

theorem junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) (total_mail : ℕ) :
  houses_per_block = 20 → mail_per_house = 32 → total_mail = 640 := by
  intros hpb_price mph_correct
  sorry

end junk_mail_per_block_l260_260214


namespace Pablo_puzzle_completion_l260_260939

theorem Pablo_puzzle_completion :
  let pieces_per_hour := 100
  let puzzles_400 := 15
  let pieces_per_puzzle_400 := 400
  let puzzles_700 := 10
  let pieces_per_puzzle_700 := 700
  let daily_work_hours := 6
  let daily_work_400_hours := 4
  let daily_work_700_hours := 2
  let break_every_hours := 2
  let break_time := 30 / 60   -- 30 minutes break in hours

  let total_pieces_400 := puzzles_400 * pieces_per_puzzle_400
  let total_pieces_700 := puzzles_700 * pieces_per_puzzle_700
  let total_pieces := total_pieces_400 + total_pieces_700

  let effective_daily_hours := daily_work_hours - (daily_work_hours / break_every_hours * break_time)
  let pieces_400_per_day := daily_work_400_hours * pieces_per_hour
  let pieces_700_per_day := (effective_daily_hours - daily_work_400_hours) * pieces_per_hour
  let total_pieces_per_day := pieces_400_per_day + pieces_700_per_day
  
  total_pieces / total_pieces_per_day = 26 := by
sorry

end Pablo_puzzle_completion_l260_260939


namespace pages_read_in_7_days_l260_260502

-- Definitions of the conditions
def total_hours : ℕ := 10
def days : ℕ := 5
def pages_per_hour : ℕ := 50
def reading_days : ℕ := 7

-- Compute intermediate steps
def hours_per_day : ℕ := total_hours / days
def pages_per_day : ℕ := pages_per_hour * hours_per_day

-- Lean statement to prove Tom reads 700 pages in 7 days
theorem pages_read_in_7_days :
  pages_per_day * reading_days = 700 :=
by
  -- We can add the intermediate steps here as sorry, as we will not do the proof
  sorry

end pages_read_in_7_days_l260_260502


namespace correct_exponentiation_l260_260511

theorem correct_exponentiation (a : ℕ) : 
  (a^3 * a^2 = a^5) ∧ ¬(a^3 + a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^10 / a^2 = a^5) :=
by
  -- Proof steps and actual mathematical validation will go here.
  -- For now, we skip the actual proof due to the problem requirements.
  sorry

end correct_exponentiation_l260_260511


namespace total_cookies_sold_l260_260401

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l260_260401


namespace complement_of_beta_l260_260284

variable (α β : ℝ)
variable (compl : α + β = 180)
variable (alpha_greater_beta : α > β)

theorem complement_of_beta (h : α + β = 180) (h' : α > β) : 90 - β = (1 / 2) * (α - β) :=
by
  sorry

end complement_of_beta_l260_260284


namespace solve_for_y_l260_260011

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l260_260011


namespace rick_ironed_27_pieces_l260_260782

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l260_260782


namespace simple_state_petya_wins_complex_state_vasya_wins_l260_260921

-- Definitions

def is_tree {V : Type} (G : simple_graph V) : Prop :=
  G.connected ∧ G.acyclic

inductive game_result
| petya_wins
| vasya_wins

-- Statements

theorem simple_state_petya_wins {V : Type} (G : simple_graph V) :
  (is_tree G) → (game_result.petya_wins) :=
by
  intros tree_prop
  sorry

theorem complex_state_vasya_wins {V : Type} (G : simple_graph V) :
  (¬ is_tree G) → (game_result.vasya_wins) :=
by
  intros not_tree_prop
  sorry

end simple_state_petya_wins_complex_state_vasya_wins_l260_260921


namespace equation_no_solution_for_k_7_l260_260137

theorem equation_no_solution_for_k_7 :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → ¬ (x ^ 2 - 1) / (x - 3) = (x ^ 2 - 7) / (x - 5) :=
by
  intro x h
  have h1 : x ≠ 3 := h.1
  have h2 : x ≠ 5 := h.2
  sorry

end equation_no_solution_for_k_7_l260_260137


namespace water_formed_from_reaction_l260_260421

-- Definitions
def mol_mass_water : ℝ := 18.015
def water_formed_grams (moles_water : ℝ) : ℝ := moles_water * mol_mass_water

-- Statement
theorem water_formed_from_reaction (moles_water : ℝ) :
  18 = water_formed_grams moles_water :=
by sorry

end water_formed_from_reaction_l260_260421


namespace blue_marbles_initial_count_l260_260903

variables (x y : ℕ)

theorem blue_marbles_initial_count (h1 : 5 * x = 8 * y) (h2 : 3 * (x - 12) = y + 21) : x = 24 :=
sorry

end blue_marbles_initial_count_l260_260903


namespace ninth_term_arithmetic_sequence_l260_260361

-- Definitions based on conditions:
def first_term : ℚ := 5 / 6
def seventeenth_term : ℚ := 5 / 8

-- Here is the main statement we need to prove:
theorem ninth_term_arithmetic_sequence : (first_term + 8 * ((seventeenth_term - first_term) / 16) = 15 / 16) :=
by
  sorry

end ninth_term_arithmetic_sequence_l260_260361


namespace bleaching_process_percentage_decrease_l260_260247

noncomputable def total_percentage_decrease (L B : ℝ) : ℝ :=
  let area1 := (0.80 * L) * (0.90 * B)
  let area2 := (0.85 * (0.80 * L)) * (0.95 * (0.90 * B))
  let area3 := (0.90 * (0.85 * (0.80 * L))) * (0.92 * (0.95 * (0.90 * B)))
  ((L * B - area3) / (L * B)) * 100

theorem bleaching_process_percentage_decrease (L B : ℝ) :
  total_percentage_decrease L B = 44.92 :=
by
  sorry

end bleaching_process_percentage_decrease_l260_260247


namespace net_profit_is_correct_l260_260801

-- Define the known quantities
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.20
def markup : ℝ := 45

-- Define the derived quantities based on the conditions
def overhead : ℝ := overhead_percentage * purchase_price
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + markup
def net_profit : ℝ := selling_price - total_cost

-- The statement to prove
theorem net_profit_is_correct : net_profit = 45 := by
  sorry

end net_profit_is_correct_l260_260801


namespace generated_surface_l260_260956

theorem generated_surface (L : ℝ → ℝ → ℝ → Prop)
  (H1 : ∀ x y z, L x y z → y = z) 
  (H2 : ∀ t, L (t^2 / 2) t 0) 
  (H3 : ∀ s, L (s^2 / 3) 0 s) : 
  ∀ y z, ∃ x, L x y z → x = (y - z) * (y / 2 - z / 3) :=
by
  sorry

end generated_surface_l260_260956


namespace problem_statement_l260_260891

theorem problem_statement (a b : ℤ) (h : |a + 5| + (b - 2) ^ 2 = 0) : (a + b) ^ 2010 = 3 ^ 2010 :=
by
  sorry

end problem_statement_l260_260891


namespace tangent_line_through_origin_eq_ex_l260_260128

theorem tangent_line_through_origin_eq_ex :
  ∃ (k : ℝ), (∀ x : ℝ, y = e^x) ∧ (∃ x₀ : ℝ, y - e^x₀ = e^x₀ * (x - x₀)) ∧ 
  (y = k * x) :=
sorry

end tangent_line_through_origin_eq_ex_l260_260128


namespace mixed_number_solution_l260_260693

noncomputable def mixed_number_problem : Prop :=
  let a := 4 + 2 / 7
  let b := 5 + 1 / 2
  let c := 3 + 1 / 3
  let d := 2 + 1 / 6
  (a * b) - (c + d) = 18 + 1 / 14

theorem mixed_number_solution : mixed_number_problem := by 
  sorry

end mixed_number_solution_l260_260693


namespace ratio_proof_l260_260731

variable (a b c d : ℚ)

-- Given conditions
axiom h1 : b / a = 3
axiom h2 : c / b = 4
axiom h3 : d = 5 * b

-- Theorem to be proved
theorem ratio_proof : (a + b + d) / (b + c + d) = 19 / 30 := 
by 
  sorry

end ratio_proof_l260_260731


namespace imaginary_part_of_complex_number_l260_260365

theorem imaginary_part_of_complex_number :
  let z := (1 + Complex.I)^2 * (2 + Complex.I)
  Complex.im z = 4 :=
by
  sorry

end imaginary_part_of_complex_number_l260_260365


namespace min_value_of_fraction_sum_l260_260175

theorem min_value_of_fraction_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 :=
sorry

end min_value_of_fraction_sum_l260_260175


namespace trapezoid_area_l260_260029

theorem trapezoid_area (A_outer A_inner : ℝ) (n : ℕ)
  (h_outer : A_outer = 36)
  (h_inner : A_inner = 4)
  (h_n : n = 4) :
  (A_outer - A_inner) / n = 8 := by
  sorry

end trapezoid_area_l260_260029


namespace greenville_height_of_boxes_l260_260660

theorem greenville_height_of_boxes:
  ∃ h : ℝ, 
    (20 * 20 * h) * (2160000 / (20 * 20 * h)) * 0.40 = 180 ∧ 
    400 * h = 2160000 / (2160000 / (20 * 20 * h)) ∧
    400 * h = 5400 ∧
    h = 12 :=
    sorry

end greenville_height_of_boxes_l260_260660


namespace conference_attendees_l260_260390

theorem conference_attendees (w m : ℕ) (h1 : w + m = 47) (h2 : 16 + (w - 1) = m) : w = 16 ∧ m = 31 :=
by
  sorry

end conference_attendees_l260_260390


namespace least_integer_value_abs_l260_260970

theorem least_integer_value_abs (x : ℤ) : 
  (∃ x : ℤ, (abs (3 * x + 5) ≤ 20) ∧ (∀ y : ℤ, (abs (3 * y + 5) ≤ 20) → x ≤ y)) ↔ x = -8 :=
by
  sorry

end least_integer_value_abs_l260_260970


namespace percentage_equivalence_l260_260907

theorem percentage_equivalence (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 :=
sorry

end percentage_equivalence_l260_260907


namespace k_ge_a_l260_260181

theorem k_ge_a (a k : ℕ) (h_pos_a : 0 < a) (h_pos_k : 0 < k) 
  (h_div : (a ^ 2 + k) ∣ (a - 1) * a * (a + 1)) : k ≥ a := 
sorry

end k_ge_a_l260_260181


namespace smallest_n_l260_260705

theorem smallest_n(vc: ℕ) (n: ℕ) : 
    (vc = 25) ∧ ∃ y o i : ℕ, ((25 * n = 10 * y) ∨ (25 * n = 18 * o) ∨ (25 * n = 20 * i)) → 
    n = 16 := by
    -- We state that given conditions should imply n = 16.
    sorry

end smallest_n_l260_260705


namespace prob_selected_first_eq_third_l260_260217

noncomputable def total_students_first := 800
noncomputable def total_students_second := 600
noncomputable def total_students_third := 500
noncomputable def selected_students_third := 25
noncomputable def prob_selected_third := selected_students_third / total_students_third

theorem prob_selected_first_eq_third :
  (selected_students_third / total_students_third = 1 / 20) →
  (prob_selected_third = 1 / 20) :=
by
  intros h
  sorry

end prob_selected_first_eq_third_l260_260217


namespace total_cost_production_l260_260073

-- Define the fixed cost and marginal cost per product as constants
def fixedCost : ℤ := 12000
def marginalCostPerProduct : ℤ := 200
def numberOfProducts : ℤ := 20

-- Define the total cost as the sum of fixed cost and total variable cost
def totalCost : ℤ := fixedCost + (marginalCostPerProduct * numberOfProducts)

-- Prove that the total cost is equal to 16000
theorem total_cost_production : totalCost = 16000 :=
by
  sorry

end total_cost_production_l260_260073


namespace apple_equation_l260_260583

-- Conditions directly from a)
def condition1 (x : ℕ) : Prop := (x - 1) % 3 = 0
def condition2 (x : ℕ) : Prop := (x + 2) % 4 = 0

theorem apple_equation (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : 
  (x - 1) / 3 = (x + 2) / 4 := 
sorry

end apple_equation_l260_260583


namespace area_enclosed_by_curves_l260_260070

noncomputable def areaBetweenCurves : ℝ :=
  ∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))

theorem area_enclosed_by_curves :
  (∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))) = (32 / 3 : ℝ) :=
by
  sorry

end area_enclosed_by_curves_l260_260070


namespace initial_card_count_l260_260674

theorem initial_card_count (r b : ℕ) (h₁ : (r : ℝ)/(r + b) = 1/4)
    (h₂ : (r : ℝ)/(r + (b + 6)) = 1/6) : r + b = 12 :=
by
  sorry

end initial_card_count_l260_260674


namespace medal_award_ways_l260_260088

-- Conditions
def sprinters : ℕ := 12
def americans : ℕ := 5
def medals : ℕ := 3 -- gold, silver, bronze

-- Question: Prove the number of ways to award medals with no more than two Americans winning is 1260.
theorem medal_award_ways :
  ∃ ways : ℕ,
    (ways = 210 + 630 + 420) ∧
    ways = 1260 :=
begin
  use 1260,
  split,
  { -- We are provided that the correct way count is the sum of 210, 630, and 420
    norm_num, 
  },
  { -- The total ways should be 1260
    refl
  }
end

end medal_award_ways_l260_260088


namespace time_to_fill_is_correct_l260_260668

-- Definitions of rates
variable (R_1 : ℚ) (R_2 : ℚ)

-- Conditions given in the problem
def rate1 := (1 : ℚ) / 8
def rate2 := (1 : ℚ) / 12

-- The resultant rate when both pipes work together
def combined_rate := rate1 + rate2

-- Calculate the time taken to fill the tank
def time_to_fill_tank := 1 / combined_rate

theorem time_to_fill_is_correct (h1 : R_1 = rate1) (h2 : R_2 = rate2) :
  time_to_fill_tank = 24 / 5 := by
  sorry

end time_to_fill_is_correct_l260_260668


namespace three_digit_ends_with_itself_iff_l260_260875

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l260_260875


namespace math_group_question_count_l260_260829

theorem math_group_question_count (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m = 6 ∧ n = 3 := 
sorry

end math_group_question_count_l260_260829


namespace circle_standard_equation_l260_260134

noncomputable def circle_equation (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 1

theorem circle_standard_equation : circle_equation 2 := by
  sorry

end circle_standard_equation_l260_260134


namespace problem_b_value_l260_260199

theorem problem_b_value (b : ℤ)
  (h1 : 0 ≤ b)
  (h2 : b ≤ 20)
  (h3 : (3 - b) % 17 = 0) : b = 3 :=
sorry

end problem_b_value_l260_260199


namespace translate_line_upwards_l260_260966

theorem translate_line_upwards (x y y' : ℝ) (h : y = -2 * x) (t : y' = y + 4) : y' = -2 * x + 4 :=
by
  sorry

end translate_line_upwards_l260_260966


namespace man_speed_l260_260834

theorem man_speed (train_length : ℝ) (time_to_cross : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) (h2 : time_to_cross = 6) (h3 : train_speed_kmph = 54.99520038396929) : 
  ∃ man_speed : ℝ, man_speed = 16.66666666666667 - 15.27644455165814 :=
by sorry

end man_speed_l260_260834


namespace sarah_flour_total_l260_260194

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l260_260194


namespace dealer_selling_price_above_cost_l260_260664

variable (cost_price : ℝ := 100)
variable (discount_percent : ℝ := 20)
variable (profit_percent : ℝ := 20)

theorem dealer_selling_price_above_cost :
  ∀ (x : ℝ), 
  (0.8 * x = 1.2 * cost_price) → 
  x = cost_price * (1 + profit_percent / 100) :=
by
  sorry

end dealer_selling_price_above_cost_l260_260664


namespace arrange_digits_multiple_of_5_l260_260458

theorem arrange_digits_multiple_of_5 : 
    let digits := [1,2,2,5],
        permutations := Multiset.perm [1,2,2],
        valid_permutations := permutations.filter (λ p, (Multiset.singleton 5 ++ p) ∈ digits.permutations) in
    valid_permutations.card = 3 :=
by
  -- formal proof would go here, for now, we assume the statement
  sorry

end arrange_digits_multiple_of_5_l260_260458


namespace Jose_played_football_l260_260764

theorem Jose_played_football :
  ∀ (total_hours : ℝ) (basketball_minutes : ℕ) (minutes_per_hour : ℕ), total_hours = 1.5 → basketball_minutes = 60 →
  (total_hours * minutes_per_hour - basketball_minutes = 30) :=
by
  intros total_hours basketball_minutes minutes_per_hour h1 h2
  sorry

end Jose_played_football_l260_260764


namespace tom_reads_700_pages_in_7_days_l260_260501

theorem tom_reads_700_pages_in_7_days
  (total_hours : ℕ)
  (total_days : ℕ)
  (pages_per_hour : ℕ)
  (reads_same_amount_every_day : Prop)
  (h1 : total_hours = 10)
  (h2 : total_days = 5)
  (h3 : pages_per_hour = 50)
  (h4 : reads_same_amount_every_day) :
  (total_hours / total_days) * (pages_per_hour * 7) = 700 :=
by
  -- Begin and skip proof with sorry
  sorry

end tom_reads_700_pages_in_7_days_l260_260501


namespace exists_root_in_interval_l260_260946

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x - 2

theorem exists_root_in_interval :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x > 0, ContinuousAt f x) → (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by sorry

end exists_root_in_interval_l260_260946


namespace find_expression_value_l260_260554

-- We declare our variables x and y
variables (x y : ℝ)

-- We state our conditions as hypotheses
def h1 : 3 * x + y = 5 := sorry
def h2 : x + 3 * y = 8 := sorry

-- We prove the given mathematical expression
theorem find_expression_value (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 10 * x^2 + 19 * x * y + 10 * y^2 = 153 := 
by
  -- We intentionally skip the proof
  sorry

end find_expression_value_l260_260554


namespace problem1_problem2_l260_260149

theorem problem1 (m : ℝ) (H : m > 0) (p : ∀ x : ℝ, (x+1)*(x-5) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) : m ≥ 4 :=
sorry

theorem problem2 (x : ℝ) (m : ℝ) (H : m = 5) (disj : ∀ x : ℝ, ((x+1)*(x-5) ≤ 0 ∨ (1 - m ≤ x ∧ x ≤ 1 + m))
) (conj : ¬ ∃ x : ℝ, (x+1)*(x-5) ≤ 0 ∧ (1 - m ≤ x ∧ x ≤ 1 + m)) : (-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6) :=
sorry

end problem1_problem2_l260_260149


namespace abs_neg_2023_l260_260794

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l260_260794


namespace smallest_n_satisfies_conditions_l260_260648

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260648


namespace equation_holds_l260_260741

-- Positive integers less than 10
def is_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

theorem equation_holds (a b c : ℕ) (ha : is_lt_10 a) (hb : is_lt_10 b) (hc : is_lt_10 c) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 :=
by
  sorry

end equation_holds_l260_260741


namespace Problem_l260_260448

theorem Problem (x y : ℝ) (h1 : 2*x + 2*y = 10) (h2 : x*y = -15) : 4*(x^2) + 4*(y^2) = 220 := 
by
  sorry

end Problem_l260_260448


namespace rental_cost_per_day_l260_260236

theorem rental_cost_per_day (p m c : ℝ) (d : ℝ) (hc : c = 0.08) (hm : m = 214.0) (hp : p = 46.12) (h_total : p = d + m * c) : d = 29.00 := 
by
  sorry

end rental_cost_per_day_l260_260236


namespace circle_equation_l260_260432

theorem circle_equation {a b c : ℝ} (hc : c ≠ 0) :
  ∃ D E F : ℝ, 
    (D = -(a + b)) ∧
    (E = - (c + ab / c)) ∧ 
    (F = ab) ∧
    ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 :=
sorry

end circle_equation_l260_260432


namespace chickens_at_stacy_farm_l260_260619
-- Importing the necessary library

-- Defining the provided conditions and correct answer in Lean 4.
theorem chickens_at_stacy_farm (C : ℕ) (piglets : ℕ) (goats : ℕ) : 
  piglets = 40 → 
  goats = 34 → 
  (C + piglets + goats) = 2 * 50 → 
  C = 26 :=
by
  intros h_piglets h_goats h_animals
  sorry

end chickens_at_stacy_farm_l260_260619


namespace max_self_intersections_polyline_7_l260_260818

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end max_self_intersections_polyline_7_l260_260818


namespace quadractic_transformation_sum_l260_260913

theorem quadractic_transformation_sum :
  let a := 5
  let h := 2
  let k := -12
  a + h + k = -5 := 
by
  sorry

end quadractic_transformation_sum_l260_260913


namespace union_set_subset_range_intersection_empty_l260_260147

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

-- Question 1: When m = -1, prove A ∪ B = { x | -2 < x < 3 }
theorem union_set (m : ℝ) (h : m = -1) : A ∪ B m = { x | -2 < x ∧ x < 3 } := by
  sorry

-- Question 2: If A ⊆ B, prove m ∈ (-∞, -2]
theorem subset_range (m : ℝ) (h : A ⊆ B m) : m ∈ Set.Iic (-2) := by
  sorry

-- Question 3: If A ∩ B = ∅, prove m ∈ [0, +∞)
theorem intersection_empty (m : ℝ) (h : A ∩ B m = ∅) : m ∈ Set.Ici 0 := by
  sorry

end union_set_subset_range_intersection_empty_l260_260147


namespace parabola_equation_l260_260994

-- Define the constants and the conditions
def parabola_focus : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ × ℝ := (3, 7, -21)

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
  a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd a b) c) d) e) f = 1 ∧
  (a : ℝ) * x^2 + (b : ℝ) * x * y + (c : ℝ) * y^2 + (d : ℝ) * x + (e : ℝ) * y + (f : ℝ) = 
  49 * x^2 - 42 * x * y + 9 * y^2 - 222 * x - 54 * y + 603 := sorry

end parabola_equation_l260_260994


namespace factorize_expression_l260_260131

theorem factorize_expression (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) :=
  sorry

end factorize_expression_l260_260131


namespace zarnin_staffing_l260_260694

open Finset

theorem zarnin_staffing :
  let total_resumes := 30
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let positions := 5
  suitable_resumes = 20 → 
  positions = 5 → 
  Nat.factorial suitable_resumes / Nat.factorial (suitable_resumes - positions) = 930240 := by
  intro total_resumes unsuitable_resumes suitable_resumes positions h1 h2
  have hs : suitable_resumes = 20 := h1
  have hp : positions = 5 := h2
  sorry

end zarnin_staffing_l260_260694


namespace leg_length_of_isosceles_right_triangle_l260_260610

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * real.sqrt 2

def median_to_hypotenuse (a b c m : ℝ) : Prop :=
  is_isosceles_right_triangle a b c ∧ m = c / 2

-- The proof problem statement
theorem leg_length_of_isosceles_right_triangle (a b c m : ℝ) (h1 : is_isosceles_right_triangle a b c)
  (h2 : median_to_hypotenuse a b c m) (h3 : m = 12) : a = 12 * real.sqrt 2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l260_260610


namespace train_speed_is_correct_l260_260366

-- Define the conditions.
def length_of_train : ℕ := 1800 -- Length of the train in meters.
def time_to_cross_platform : ℕ := 60 -- Time to cross the platform in seconds (1 minute).

-- Define the statement that needs to be proved.
def speed_of_train : ℕ := (2 * length_of_train) / time_to_cross_platform

-- State the theorem.
theorem train_speed_is_correct :
  speed_of_train = 60 := by
  sorry -- Proof is not required.

end train_speed_is_correct_l260_260366


namespace three_digit_ends_with_itself_iff_l260_260876

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l260_260876


namespace fraction_quaduple_l260_260104

variable (b a : ℤ)

theorem fraction_quaduple (h₁ : a ≠ 0) : (2 * b) / (a / 2) = 4 * (b / a) :=
by
  sorry

end fraction_quaduple_l260_260104


namespace solve_for_y_l260_260012

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l260_260012


namespace minimize_quadratic_l260_260225

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end minimize_quadratic_l260_260225


namespace calc_root_diff_l260_260844

theorem calc_root_diff : 81^(1/4) - 16^(1/2) = -1 := by
  sorry

end calc_root_diff_l260_260844


namespace find_f_l260_260230

theorem find_f (f : ℕ → ℕ) :
  (∀ a b c : ℕ, ((f a + f b + f c) - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) →
  (∀ n : ℕ, f n = n * n) :=
sorry

end find_f_l260_260230


namespace problem_part1_problem_part2_l260_260896

-- Define what we need to prove
theorem problem_part1 (x : ℝ) (a b : ℤ) 
  (h : (2*x - 21)*(3*x - 7) - (3*x - 7)*(x - 13) = (3*x + a)*(x + b)): 
  a + 3*b = -31 := 
by {
  -- We know from the problem that h holds,
  -- thus the values of a and b must satisfy the condition.
  sorry
}

theorem problem_part2 (x : ℝ) : 
  (x^2 - 3*x + 2) = (x - 1)*(x - 2) := 
by {
  sorry
}

end problem_part1_problem_part2_l260_260896


namespace integer_solution_for_x_l260_260274

theorem integer_solution_for_x (x : ℤ) : 
  (∃ y z : ℤ, x = 7 * y + 3 ∧ x = 5 * z + 2) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by
  sorry

end integer_solution_for_x_l260_260274


namespace no_such_functions_exist_l260_260702

theorem no_such_functions_exist :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 :=
by
  sorry

end no_such_functions_exist_l260_260702


namespace tan_theta_perpendicular_vectors_l260_260296

theorem tan_theta_perpendicular_vectors (θ : ℝ) (h : Real.sqrt 3 * Real.cos θ + Real.sin θ = 0) : Real.tan θ = - Real.sqrt 3 :=
sorry

end tan_theta_perpendicular_vectors_l260_260296


namespace product_is_even_l260_260354

theorem product_is_even (a b c : ℤ) : Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end product_is_even_l260_260354


namespace chef_sold_12_meals_l260_260752

theorem chef_sold_12_meals
  (initial_meals_lunch : ℕ)
  (additional_meals_dinner : ℕ)
  (meals_left_after_lunch : ℕ)
  (meals_for_dinner : ℕ)
  (H1 : initial_meals_lunch = 17)
  (H2 : additional_meals_dinner = 5)
  (H3 : meals_for_dinner = 10) :
  ∃ (meals_sold_lunch : ℕ), meals_sold_lunch = 12 := by
  sorry

end chef_sold_12_meals_l260_260752


namespace largest_common_term_in_range_1_to_200_l260_260688

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end largest_common_term_in_range_1_to_200_l260_260688


namespace remaining_credit_l260_260591

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l260_260591


namespace lines_are_coplanar_l260_260698

-- Define the first line
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, 2 - m * t, 6 + t)

-- Define the second line
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ :=
  (4 + m * u, 5 + 3 * u, 8 + 2 * u)

-- Define the vector connecting points on the lines when t=0 and u=0
def connecting_vector : ℝ × ℝ × ℝ :=
  (1, 3, 2)

-- Define the cross product of the direction vectors
def cross_product (m : ℝ) : ℝ × ℝ × ℝ :=
  ((-2 * m - 3), (m + 2), (6 + 2 * m))

-- Prove that lines are coplanar when m = -9/4
theorem lines_are_coplanar : ∃ k : ℝ, ∀ m : ℝ,
  cross_product m = (k * 1, k * 3, k * 2) → m = -9/4 :=
by
  sorry

end lines_are_coplanar_l260_260698


namespace winner_lifted_weight_l260_260811

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l260_260811


namespace monomial_degree_and_coefficient_l260_260003

theorem monomial_degree_and_coefficient (a b : ℤ) (h1 : -a = 7) (h2 : 1 + b = 4) : a + b = -4 :=
by
  sorry

end monomial_degree_and_coefficient_l260_260003


namespace find_g_3_l260_260362

def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 3) : g 3 = 0 := 
by
  sorry

end find_g_3_l260_260362


namespace k_equals_10_l260_260084

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) : ℕ → α
  | 0     => a
  | (n+1) => a + (n+1) * d

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem k_equals_10
  (a d : α)
  (h1 : sum_of_first_n_terms a d 9 = sum_of_first_n_terms a d 4)
  (h2 : arithmetic_sequence a d 4 + arithmetic_sequence a d 10 = 0) :
  k = 10 :=
sorry

end k_equals_10_l260_260084


namespace cookies_sold_by_Lucy_l260_260253

theorem cookies_sold_by_Lucy :
  let cookies_first_round := 34
  let cookies_second_round := 27
  cookies_first_round + cookies_second_round = 61 := by
  sorry

end cookies_sold_by_Lucy_l260_260253


namespace find_c_l260_260474

-- Define the functions p and q as given in the conditions
def p (x : ℝ) : ℝ := 3 * x - 9
def q (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

-- State the main theorem with conditions and goal
theorem find_c (c : ℝ) (h : p (q 3 c) = 15) : c = 4 := by
  sorry -- Proof is not required

end find_c_l260_260474


namespace area_percentage_increase_l260_260795

theorem area_percentage_increase (r1 r2 : ℝ) (π : ℝ) (area1 area2 : ℝ) (N : ℝ) :
  r1 = 6 → r2 = 4 → area1 = π * r1 ^ 2 → area2 = π * r2 ^ 2 →
  N = 125 →
  ((area1 - area2) / area2) * 100 = N :=
by {
  sorry
}

end area_percentage_increase_l260_260795


namespace probability_of_drawing_white_ball_probability_with_additional_white_balls_l260_260457

noncomputable def total_balls := 6 + 9 + 3
noncomputable def initial_white_balls := 3

theorem probability_of_drawing_white_ball :
  (initial_white_balls : ℚ) / (total_balls : ℚ) = 1 / 6 :=
sorry

noncomputable def additional_white_balls_needed := 2

theorem probability_with_additional_white_balls :
  (initial_white_balls + additional_white_balls_needed : ℚ) / (total_balls + additional_white_balls_needed : ℚ) = 1 / 4 :=
sorry

end probability_of_drawing_white_ball_probability_with_additional_white_balls_l260_260457


namespace distinct_positive_integer_quadruples_l260_260151

theorem distinct_positive_integer_quadruples 
  (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a + b = c * d) (h8 : a * b = c + d) :
  (a, b, c, d) = (1, 5, 2, 3)
  ∨ (a, b, c, d) = (1, 5, 3, 2)
  ∨ (a, b, c, d) = (5, 1, 2, 3)
  ∨ (a, b, c, d) = (5, 1, 3, 2)
  ∨ (a, b, c, d) = (2, 3, 1, 5)
  ∨ (a, b, c, d) = (2, 3, 5, 1)
  ∨ (a, b, c, d) = (3, 2, 1, 5)
  ∨ (a, b, c, d) = (3, 2, 5, 1) :=
  sorry

end distinct_positive_integer_quadruples_l260_260151


namespace kaleb_cherries_left_l260_260926

theorem kaleb_cherries_left (initial_cherries eaten_cherries remaining_cherries : ℕ) (h1 : initial_cherries = 67) (h2 : eaten_cherries = 25) : remaining_cherries = initial_cherries - eaten_cherries → remaining_cherries = 42 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end kaleb_cherries_left_l260_260926


namespace arkansas_tshirts_sold_l260_260945

theorem arkansas_tshirts_sold (A T : ℕ) (h1 : A + T = 163) (h2 : 98 * A = 8722) : A = 89 := by
  -- We state the problem and add 'sorry' to skip the actual proof
  sorry

end arkansas_tshirts_sold_l260_260945


namespace equation_is_hyperbola_l260_260701

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

-- Theorem stating that the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ x y : ℝ, equation x y → (∃ A B : ℝ, A * x^2 - B * y^2 = 1) :=
by
  sorry

end equation_is_hyperbola_l260_260701


namespace odd_prime_power_condition_l260_260097

noncomputable def is_power_of (a b : ℕ) : Prop :=
  ∃ t : ℕ, b = a ^ t

theorem odd_prime_power_condition (n p x y k : ℕ) (hn : 1 < n) (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hx_odd : x % 2 ≠ 0) 
  (hy_odd : y % 2 ≠ 0) (h_eq : x^n + y^n = p^k) :
  is_power_of p n :=
sorry

end odd_prime_power_condition_l260_260097


namespace jessica_not_work_days_l260_260763

theorem jessica_not_work_days:
  ∃ (x y z : ℕ), 
    (x + y + z = 30) ∧
    (80 * x - 40 * y + 40 * z = 1600) ∧
    (z = 5) ∧
    (y = 5) :=
by
  sorry

end jessica_not_work_days_l260_260763


namespace second_horse_revolutions_l260_260240

-- Define the parameters and conditions:
def r₁ : ℝ := 30  -- Distance of the first horse from the center
def revolutions₁ : ℕ := 15  -- Number of revolutions by the first horse
def r₂ : ℝ := 5  -- Distance of the second horse from the center

-- Define the statement to prove:
theorem second_horse_revolutions : r₂ * (↑revolutions₁ * r₁⁻¹) * (↑revolutions₁) = 90 := 
by sorry

end second_horse_revolutions_l260_260240


namespace find_speed_of_man_l260_260106

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
(v_m + v_s = 6) ∧ (v_m - v_s = 8)

theorem find_speed_of_man :
  ∃ v_m v_s : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 7 :=
by
  sorry

end find_speed_of_man_l260_260106


namespace probability_on_hyperbola_l260_260055

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l260_260055


namespace nabla_four_seven_l260_260398

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_four_seven : nabla 4 7 = 11 / 29 :=
by
  sorry

end nabla_four_seven_l260_260398


namespace smallest_n_satisfies_conditions_l260_260625

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260625


namespace quadratic_perfect_square_form_l260_260961

def quadratic_is_perfect_square (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

theorem quadratic_perfect_square_form (a b c : ℤ) (h : quadratic_is_perfect_square a b c) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
  sorry

end quadratic_perfect_square_form_l260_260961


namespace solve_for_y_l260_260013

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l260_260013


namespace basketball_club_members_l260_260751

theorem basketball_club_members :
  let sock_cost := 6
  let tshirt_additional_cost := 8
  let total_cost := 4440
  let cost_per_member := sock_cost + 2 * (sock_cost + tshirt_additional_cost)
  total_cost / cost_per_member = 130 :=
by
  sorry

end basketball_club_members_l260_260751


namespace pugs_working_together_l260_260237

theorem pugs_working_together (P : ℕ) (H1 : P * 45 = 15 * 12) : P = 4 :=
by {
  sorry
}

end pugs_working_together_l260_260237


namespace total_tissues_brought_l260_260952

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l260_260952


namespace sale_price_relative_to_original_l260_260219

variable (x : ℝ)

def increased_price (x : ℝ) := 1.30 * x
def sale_price (increased_price : ℝ) := 0.90 * increased_price

theorem sale_price_relative_to_original (x : ℝ) :
  sale_price (increased_price x) = 1.17 * x :=
by
  sorry

end sale_price_relative_to_original_l260_260219


namespace combined_weight_of_boxes_l260_260924

def weight_box1 : ℝ := 2
def weight_box2 : ℝ := 11
def weight_box3 : ℝ := 5

theorem combined_weight_of_boxes : weight_box1 + weight_box2 + weight_box3 = 18 := by
  sorry

end combined_weight_of_boxes_l260_260924


namespace ironed_clothing_l260_260780

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l260_260780


namespace fraction_product_l260_260255

theorem fraction_product :
  ((5/4) * (8/16) * (20/12) * (32/64) * (50/20) * (40/80) * (70/28) * (48/96) : ℚ) = 625/768 := 
by
  sorry

end fraction_product_l260_260255


namespace coffee_serving_time_between_1_and_2_is_correct_l260_260696

theorem coffee_serving_time_between_1_and_2_is_correct
    (x : ℝ)
    (h_pos: 0 < x)
    (h_lt: x < 60) :
    30 + (x / 2) = 360 - (6 * x) → x = 660 / 13 :=
by
  sorry

end coffee_serving_time_between_1_and_2_is_correct_l260_260696


namespace green_peaches_more_than_red_l260_260962

theorem green_peaches_more_than_red :
  let red_peaches := 5
  let green_peaches := 11
  (green_peaches - red_peaches) = 6 := by
  sorry

end green_peaches_more_than_red_l260_260962


namespace divide_0_24_by_0_004_l260_260846

theorem divide_0_24_by_0_004 : 0.24 / 0.004 = 60 := by
  sorry

end divide_0_24_by_0_004_l260_260846


namespace pages_read_in_7_days_l260_260503

-- Definitions of the conditions
def total_hours : ℕ := 10
def days : ℕ := 5
def pages_per_hour : ℕ := 50
def reading_days : ℕ := 7

-- Compute intermediate steps
def hours_per_day : ℕ := total_hours / days
def pages_per_day : ℕ := pages_per_hour * hours_per_day

-- Lean statement to prove Tom reads 700 pages in 7 days
theorem pages_read_in_7_days :
  pages_per_day * reading_days = 700 :=
by
  -- We can add the intermediate steps here as sorry, as we will not do the proof
  sorry

end pages_read_in_7_days_l260_260503


namespace ball_count_l260_260307

theorem ball_count (r b y : ℕ) 
  (h1 : b + y = 9) 
  (h2 : r + y = 5) 
  (h3 : r + b = 6) : 
  r + b + y = 10 := 
  sorry

end ball_count_l260_260307


namespace triangle_area_correct_l260_260222

def line1 (x : ℝ) : ℝ := 8
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the intersection points
def intersection1 : ℝ × ℝ := (6, line1 6)
def intersection2 : ℝ × ℝ := (-6, line1 (-6))
def intersection3 : ℝ × ℝ := (0, line2 0)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_correct :
  triangle_area intersection1 intersection2 intersection3 = 36 :=
by
  sorry

end triangle_area_correct_l260_260222


namespace seating_arrangements_equal_600_l260_260298

-- Definitions based on the problem conditions
def number_of_people : Nat := 4
def number_of_chairs : Nat := 8
def consecutive_empty_seats : Nat := 3

-- Theorem statement
theorem seating_arrangements_equal_600
  (h_people : number_of_people = 4)
  (h_chairs : number_of_chairs = 8)
  (h_consecutive_empty_seats : consecutive_empty_seats = 3) :
  (∃ (arrangements : Nat), arrangements = 600) :=
sorry

end seating_arrangements_equal_600_l260_260298


namespace find_c_plus_d_l260_260135

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
if x < 3 then c * x + d else 10 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x, g c d (g c d x) = x) : c + d = 4.5 :=
sorry

end find_c_plus_d_l260_260135


namespace tom_made_washing_cars_l260_260765

-- Definitions of the conditions
def initial_amount : ℕ := 74
def final_amount : ℕ := 86

-- Statement to be proved
theorem tom_made_washing_cars : final_amount - initial_amount = 12 := by
  sorry

end tom_made_washing_cars_l260_260765


namespace eval_sqrt_pow_l260_260411

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l260_260411


namespace fraction_spent_on_sandwich_l260_260463
    
theorem fraction_spent_on_sandwich 
  (x : ℚ)
  (h1 : 90 * x + 90 * (1/6) + 90 * (1/2) + 12 = 90) : 
  x = 1/5 :=
by
  sorry

end fraction_spent_on_sandwich_l260_260463


namespace total_cookies_sold_l260_260402

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l260_260402


namespace slant_heights_of_cones_l260_260443

-- Define the initial conditions
variables (r r1 x y : Real)

-- Define the surface area condition
def surface_area_condition : Prop :=
  r * Real.sqrt (r ^ 2 + x ^ 2) + r ^ 2 = r1 * Real.sqrt (r1 ^ 2 + y ^ 2) + r1 ^ 2

-- Define the volume condition
def volume_condition : Prop :=
  r ^ 2 * Real.sqrt (x ^ 2 - r ^ 2) = r1 ^ 2 * Real.sqrt (y ^ 2 - r1 ^ 2)

-- Statement of the proof problem: Prove that the slant heights x and y are given by
theorem slant_heights_of_cones
  (h1 : surface_area_condition r r1 x y)
  (h2 : volume_condition r r1 x y) :
  x = (r ^ 2 + 2 * r1 ^ 2) / r ∧ y = (r1 ^ 2 + 2 * r ^ 2) / r1 := 
  sorry

end slant_heights_of_cones_l260_260443


namespace geometric_sequence_condition_l260_260453

theorem geometric_sequence_condition {a : ℕ → ℝ} (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) : 
  (a 3 * a 5 = 16) ↔ a 4 = 4 :=
sorry

end geometric_sequence_condition_l260_260453


namespace problem1_problem2_l260_260336

theorem problem1 (x : ℝ) (a : ℝ) (h : a = 1) (hp : a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) : 2 < x ∧ x < 3 := 
by
  sorry

theorem problem2 (x : ℝ) (a : ℝ) (hp : 0 < a ∧ a < x ∧ x < 3 * a) (hq : 2 < x ∧ x < 3) (hsuff : ∀ (a x : ℝ), (2 < x ∧ x < 3) → a < x ∧ x < 3 * a) : 1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end problem1_problem2_l260_260336


namespace negation_prop_l260_260076

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l260_260076


namespace smallest_n_l260_260658

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260658


namespace geometry_biology_overlap_diff_l260_260383

theorem geometry_biology_overlap_diff :
  ∀ (total_students geometry_students biology_students : ℕ),
  total_students = 232 →
  geometry_students = 144 →
  biology_students = 119 →
  (max geometry_students biology_students - max 0 (geometry_students + biology_students - total_students)) = 88 :=
by
  intros total_students geometry_students biology_students
  sorry

end geometry_biology_overlap_diff_l260_260383


namespace smallest_n_l260_260657

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260657


namespace total_tissues_brought_l260_260955

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l260_260955


namespace inequality_holds_iff_m_lt_2_l260_260493

theorem inequality_holds_iff_m_lt_2 :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → x^2 - m * x + m > 0) ↔ m < 2 :=
by
  sorry

end inequality_holds_iff_m_lt_2_l260_260493


namespace homes_distance_is_65_l260_260186

noncomputable def distance_between_homes
  (maxwell_speed : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (time : ℕ) : ℕ :=
  maxwell_distance + brad_speed * time

theorem homes_distance_is_65
  (maxwell_speed : ℕ := 2)
  (brad_speed : ℕ := 3)
  (maxwell_distance : ℕ := 26)
  (time : ℕ := maxwell_distance / maxwell_speed) :
  distance_between_homes maxwell_speed brad_speed maxwell_distance time = 65 :=
by 
  sorry

end homes_distance_is_65_l260_260186


namespace workers_new_daily_wage_l260_260837

def wage_before : ℝ := 25
def increase_percentage : ℝ := 0.40

theorem workers_new_daily_wage : wage_before * (1 + increase_percentage) = 35 :=
by
  -- sorry will be replaced by the actual proof steps
  sorry

end workers_new_daily_wage_l260_260837


namespace problem1_problem2_problem3_problem4_l260_260672

-- Definitions of conversion rates used in the conditions
def sq_m_to_sq_dm : Nat := 100
def hectare_to_sq_m : Nat := 10000
def sq_cm_to_sq_dm_div : Nat := 100
def sq_km_to_hectare : Nat := 100

-- The problem statement with the expected values
theorem problem1 : 3 * sq_m_to_sq_dm = 300 := by
  sorry

theorem problem2 : 2 * hectare_to_sq_m = 20000 := by
  sorry

theorem problem3 : 5000 / sq_cm_to_sq_dm_div = 50 := by
  sorry

theorem problem4 : 8 * sq_km_to_hectare = 800 := by
  sorry

end problem1_problem2_problem3_problem4_l260_260672


namespace angle_between_line_and_plane_correct_l260_260445

variables (m n : Vector3)

noncomputable def angle_between_line_and_plane (m n : Vector3) : ℝ :=
  let cos_theta := -1 / 2
  -- The complementary angle to the one given by the cosine value
  (real.arccos cos_theta)

theorem angle_between_line_and_plane_correct
  (m n : Vector3)
  (h : real.arccos ((m.dot n) / (m.norm * n.norm)) = real.arccos (-1 / 2)) :
  angle_between_line_and_plane m n = real.arccos (-1 / 2) :=
by
  -- Proof is skipped
  sorry

end angle_between_line_and_plane_correct_l260_260445


namespace decompose_series_l260_260124

-- Define the 11-arithmetic Fibonacci sequence using the given series
def Φ₁₁₀ (n : ℕ) : ℕ :=
  if n % 11 = 0 then 0 else
  if n % 11 = 1 then 1 else
  if n % 11 = 2 then 1 else
  if n % 11 = 3 then 2 else
  if n % 11 = 4 then 3 else
  if n % 11 = 5 then 5 else
  if n % 11 = 6 then 8 else
  if n % 11 = 7 then 2 else
  if n % 11 = 8 then 10 else
  if n % 11 = 9 then 1 else
  0

-- Define the two geometric progressions
def G₁ (n : ℕ) : ℤ := 3 * (8 ^ n)
def G₂ (n : ℕ) : ℤ := 8 * (4 ^ n)

-- The decomposed sequence
def decomposedSequence (n : ℕ) : ℤ := G₁ n + G₂ n

-- The theorem to prove the decomposition
theorem decompose_series : ∀ n : ℕ, Φ₁₁₀ n = decomposedSequence n := by
  sorry

end decompose_series_l260_260124


namespace relationship_among_a_b_c_l260_260718

theorem relationship_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (4 : ℝ) ^ (1 / 2))
  (hb : b = (2 : ℝ) ^ (1 / 3))
  (hc : c = (5 : ℝ) ^ (1 / 2))
: b < a ∧ a < c := 
sorry

end relationship_among_a_b_c_l260_260718


namespace total_teeth_cleaned_l260_260506

/-
  Given:
   1. Dogs have 42 teeth.
   2. Cats have 30 teeth.
   3. Pigs have 28 teeth.
   4. There are 5 dogs.
   5. There are 10 cats.
   6. There are 7 pigs.
  Prove: The total number of teeth Vann will clean today is 706.
-/

theorem total_teeth_cleaned :
  let dogs: Nat := 5
  let cats: Nat := 10
  let pigs: Nat := 7
  let dog_teeth: Nat := 42
  let cat_teeth: Nat := 30
  let pig_teeth: Nat := 28
  (dogs * dog_teeth) + (cats * cat_teeth) + (pigs * pig_teeth) = 706 := by
  -- Proof goes here
  sorry

end total_teeth_cleaned_l260_260506


namespace isosceles_right_triangle_leg_length_l260_260608

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l260_260608


namespace quadractic_transformation_sum_l260_260914

theorem quadractic_transformation_sum :
  let a := 5
  let h := 2
  let k := -12
  a + h + k = -5 := 
by
  sorry

end quadractic_transformation_sum_l260_260914


namespace find_x_plus_y_l260_260039

theorem find_x_plus_y (x y : ℕ) 
  (h1 : 4^x = 16^(y + 1)) 
  (h2 : 5^(2 * y) = 25^(x - 2)) : 
  x + y = 2 := 
sorry

end find_x_plus_y_l260_260039


namespace set_intersection_and_polynomial_solution_l260_260901

theorem set_intersection_and_polynomial_solution {a b : ℝ} :
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  (A ∩ B = {x | x < -3}) ∧ ((A ∪ B = {x | x < -2 ∨ x > 1}) →
    (a = 2 ∧ b = -4)) :=
by
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  sorry

end set_intersection_and_polynomial_solution_l260_260901


namespace find_roots_of_quadratic_l260_260424

open Complex

theorem find_roots_of_quadratic : 
  let z1 := sqrt 7 - 1 + (sqrt 7 / 2) * Complex.I
      z2 := -sqrt 7 - 1 - (sqrt 7 / 2) * Complex.I in
    (∀ z : ℂ, z^2 + 2 * z = 3 + 7 * Complex.I ↔ (z = z1 ∨ z = z2))
:= sorry

end find_roots_of_quadratic_l260_260424


namespace rocket_parachute_opens_l260_260679

theorem rocket_parachute_opens (h t : ℝ) : h = -t^2 + 12 * t + 1 ∧ h = 37 -> t = 6 :=
by sorry

end rocket_parachute_opens_l260_260679


namespace atomic_weight_S_is_correct_l260_260422

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end atomic_weight_S_is_correct_l260_260422


namespace balls_in_boxes_l260_260569

theorem balls_in_boxes :
  (∑ k in finset.range 4, nat.choose 7 k) = 64 :=
by
  sorry

end balls_in_boxes_l260_260569


namespace common_difference_is_minus_two_l260_260371

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
noncomputable def sum_arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem common_difference_is_minus_two
  (a1 d : ℤ)
  (h1 : sum_arith_seq a1 d 5 = 15)
  (h2 : arith_seq a1 d 2 = 5) :
  d = -2 :=
by
  sorry

end common_difference_is_minus_two_l260_260371


namespace num_values_of_n_l260_260697

theorem num_values_of_n (a b c : ℕ) (h : 7 * a + 77 * b + 7777 * c = 8000) : 
  ∃ n : ℕ, (n = a + 2 * b + 4 * c) ∧ (110 * n ≤ 114300) ∧ ((8000 - 7 * a) % 70 = 7 * (10 * b + 111 * c) % 70) := 
sorry

end num_values_of_n_l260_260697


namespace find_a_b_l260_260485

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_b 
  (h_max : ∀ x, f a b x ≤ 3)
  (h_min : ∀ x, f a b x ≥ 2)
  : (a = 0.5 ∨ a = -0.5) ∧ b = 2.5 :=
by
  sorry

end find_a_b_l260_260485


namespace coefficient_of_x4_l260_260437

theorem coefficient_of_x4 (n : ℕ) (f : ℕ → ℕ → ℝ)
  (h1 : (2 : ℕ) ^ n = 256) :
  (f 8 4) * (2 : ℕ) ^ 4 = 1120 :=
by
  sorry

end coefficient_of_x4_l260_260437


namespace negation_of_proposition_l260_260079

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition_l260_260079


namespace statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l260_260825

-- Definitions of conditions
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ)
  (angles : Fin n → ℝ)

def circumscribed (P : Polygon n) : Prop := sorry -- Definition of circumscribed
def inscribed (P : Polygon n) : Prop := sorry -- Definition of inscribed
def equal_sides (P : Polygon n) : Prop := ∀ i j, P.sides i = P.sides j
def equal_angles (P : Polygon n) : Prop := ∀ i j, P.angles i = P.angles j

-- The statements to be proved
theorem statement_I : ∀ P : Polygon n, circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_II : ∃ P : Polygon n, inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_III : ∃ P : Polygon n, circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_IV : ∀ P : Polygon n, inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_V : ∀ (P : Polygon 5), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VI : ∀ (P : Polygon 6), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VII : ∀ (P : Polygon 5), inscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VIII : ∃ (P : Polygon 6), inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_IX : ∀ (P : Polygon 5), circumscribed P → equal_angles P → equal_sides P := sorry

theorem statement_X : ∃ (P : Polygon 6), circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_XI : ∀ (P : Polygon 5), inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_XII : ∀ (P : Polygon 6), inscribed P → equal_angles P → equal_sides P := sorry

end statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l260_260825


namespace probability_on_hyperbola_l260_260054

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l260_260054


namespace edward_can_buy_candies_l260_260095

theorem edward_can_buy_candies (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ)
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 :=
by
  sorry

end edward_can_buy_candies_l260_260095


namespace cost_price_of_article_l260_260670

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 :=
by sorry

end cost_price_of_article_l260_260670


namespace sample_size_proof_l260_260984

theorem sample_size_proof (p : ℝ) (N : ℤ) (n : ℤ) (h1 : N = 200) (h2 : p = 0.25) : n = 50 :=
by
  sorry

end sample_size_proof_l260_260984


namespace cindy_first_to_get_five_l260_260839

def probability_of_five : ℚ := 1 / 6

def anne_turn (p: ℚ) : ℚ := 1 - p
def cindy_turn (p: ℚ) : ℚ := p
def none_get_five (p: ℚ) : ℚ := (1 - p)^3

theorem cindy_first_to_get_five : 
    (∑' n, (anne_turn probability_of_five * none_get_five probability_of_five ^ n) * 
                cindy_turn probability_of_five) = 30 / 91 := by 
    sorry

end cindy_first_to_get_five_l260_260839


namespace two_digit_numbers_with_5_as_second_last_digit_l260_260133

theorem two_digit_numbers_with_5_as_second_last_digit:
  ∀ N : ℕ, (10 ≤ N ∧ N ≤ 99) → (∃ k : ℤ, (N * k) % 100 / 10 = 5) ↔ ¬(N % 20 = 0) :=
by
  sorry

end two_digit_numbers_with_5_as_second_last_digit_l260_260133


namespace Lily_balls_is_3_l260_260396

-- Definitions from conditions
variable (L : ℕ)

def Frodo_balls := L + 8
def Brian_balls := 2 * (L + 8)

axiom Brian_has_22 : Brian_balls L = 22

-- The goal is to prove that Lily has 3 tennis balls
theorem Lily_balls_is_3 : L = 3 :=
by
  sorry

end Lily_balls_is_3_l260_260396


namespace volume_of_new_pyramid_l260_260109

theorem volume_of_new_pyramid (l w h : ℝ) (h_vol : (1 / 3) * l * w * h = 80) :
  (1 / 3) * (3 * l) * w * (1.8 * h) = 432 :=
by
  sorry

end volume_of_new_pyramid_l260_260109


namespace find_length_AX_l260_260920

theorem find_length_AX 
  (A B C X : Type)
  (BC BX AC : ℝ)
  (h_BC : BC = 36)
  (h_BX : BX = 30)
  (h_AC : AC = 27)
  (h_bisector : ∃ (x : ℝ), x = BX / BC ∧ x = AX / AC ) :
  ∃ AX : ℝ, AX = 22.5 := 
sorry

end find_length_AX_l260_260920


namespace gum_sharing_l260_260037

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l260_260037


namespace mean_properties_l260_260483

theorem mean_properties (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (arith_mean : (x + y + z) / 3 = 10)
  (geom_mean : (x * y * z) ^ (1 / 3) = 6)
  (harm_mean : 3 / (1/x + 1/y + 1/z) = 2.5) :
  x^2 + y^2 + z^2 = 540 := 
sorry

end mean_properties_l260_260483


namespace sin_cos_15_degrees_proof_l260_260101

noncomputable
def sin_cos_15_degrees : Prop := (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4)

theorem sin_cos_15_degrees_proof : sin_cos_15_degrees :=
by
  sorry

end sin_cos_15_degrees_proof_l260_260101


namespace velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l260_260107

noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)
noncomputable def v (A ω t : ℝ) : ℝ := deriv (x A ω) t
noncomputable def α (A ω t : ℝ) : ℝ := deriv (v A ω) t

theorem velocity_at_specific_time (A ω : ℝ) : 
  v A ω (2 * Real.pi / ω) = A * ω := 
sorry

theorem acceleration_at_specific_time (A ω : ℝ) :
  α A ω (2 * Real.pi / ω) = 0 :=
sorry

theorem acceleration_proportional_to_displacement (A ω t : ℝ) :
  α A ω t = -ω^2 * x A ω t :=
sorry

end velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l260_260107


namespace samantha_probability_l260_260599

noncomputable def probability_of_selecting_yellow_apples 
  (total_apples : ℕ) (yellow_apples : ℕ) (selection_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_apples selection_size
  let yellow_ways := Nat.choose yellow_apples selection_size
  yellow_ways / total_ways

theorem samantha_probability : 
  probability_of_selecting_yellow_apples 10 5 3 = 1 / 12 := 
by 
  sorry

end samantha_probability_l260_260599


namespace exists_large_natural_with_high_digit_sum_l260_260352

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem exists_large_natural_with_high_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10 ^ 100 :=
by sorry

end exists_large_natural_with_high_digit_sum_l260_260352


namespace find_initial_workers_l260_260985

-- Define the initial number of workers.
def initial_workers (W : ℕ) (A : ℕ) : Prop :=
  -- Condition 1: W workers can complete work A in 25 days.
  ( W * 25 = A )  ∧
  -- Condition 2: (W + 10) workers can complete work A in 15 days.
  ( (W + 10) * 15 = A )

-- The theorem states that given the conditions, the initial number of workers is 15.
theorem find_initial_workers {W A : ℕ} (h : initial_workers W A) : W = 15 :=
  sorry

end find_initial_workers_l260_260985


namespace sum_first_10_terms_l260_260292

variable (a : ℕ → ℕ)

def condition (p q : ℕ) : Prop :=
  p + q = 11 ∧ p < q

axiom condition_a_p_a_q : ∀ (p q : ℕ), (condition p q) → (a p + a q = 2^p)

theorem sum_first_10_terms (a : ℕ → ℕ) (h : ∀ (p q : ℕ), condition p q → a p + a q = 2^p) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 62) :=
by 
  sorry

end sum_first_10_terms_l260_260292


namespace evaluate_expression_l260_260363

noncomputable def f : ℝ → ℝ := sorry

lemma f_condition (a : ℝ) : f (a + 1) = f a * f 1 := sorry

lemma f_one : f 1 = 2 := sorry

theorem evaluate_expression :
  (f 2018 / f 2017) + (f 2019 / f 2018) + (f 2020 / f 2019) = 6 :=
sorry

end evaluate_expression_l260_260363


namespace basketball_weight_l260_260262

variable {b c : ℝ}

theorem basketball_weight (h1 : 8 * b = 4 * c) (h2 : 3 * c = 120) : b = 20 :=
by
  -- Proof omitted
  sorry

end basketball_weight_l260_260262


namespace ironed_clothing_l260_260779

theorem ironed_clothing (shirts_rate pants_rate shirts_hours pants_hours : ℕ)
    (h1 : shirts_rate = 4)
    (h2 : pants_rate = 3)
    (h3 : shirts_hours = 3)
    (h4 : pants_hours = 5) :
    shirts_rate * shirts_hours + pants_rate * pants_hours = 27 := by
  sorry

end ironed_clothing_l260_260779


namespace bowling_ball_weight_l260_260593

theorem bowling_ball_weight (b c : ℝ) (h1 : 9 * b = 6 * c) (h2 : 4 * c = 120) : b = 20 :=
sorry

end bowling_ball_weight_l260_260593


namespace probability_point_on_hyperbola_l260_260058

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l260_260058


namespace second_number_exists_l260_260993

theorem second_number_exists (x : ℕ) (h : 150 / x = 15) : x = 10 :=
sorry

end second_number_exists_l260_260993


namespace crayons_loss_l260_260190

def initial_crayons : ℕ := 479
def final_crayons : ℕ := 134
def crayons_lost : ℕ := initial_crayons - final_crayons

theorem crayons_loss :
  crayons_lost = 345 := by
  sorry

end crayons_loss_l260_260190


namespace inequality_proof_l260_260495

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (ab / Real.sqrt (c^2 + 3)) + (bc / Real.sqrt (a^2 + 3)) + (ca / Real.sqrt (b^2 + 3)) ≤ 3 / 2 :=
by
  sorry

end inequality_proof_l260_260495


namespace all_inequalities_hold_l260_260733

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x^2 < a^2
def condition2 : Prop := y^2 < b^2
def condition3 : Prop := z^2 < c^2

-- Inequalities to prove
def inequality1 : Prop := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a^2 * b^2 + b^2 * c^2 + c^2 * a^2
def inequality2 : Prop := x^4 + y^4 + z^4 < a^4 + b^4 + c^4
def inequality3 : Prop := x^2 * y^2 * z^2 < a^2 * b^2 * c^2

theorem all_inequalities_hold (h1 : condition1 a x) (h2 : condition2 b y) (h3 : condition3 c z) :
  inequality1 a b c x y z ∧ inequality2 a b c x y z ∧ inequality3 a b c x y z := by
  sorry

end all_inequalities_hold_l260_260733


namespace like_terms_ratio_l260_260904

theorem like_terms_ratio (m n : ℕ) (h₁ : m - 2 = 2) (h₂ : 3 = 2 * n - 1) : m / n = 2 := 
by
  sorry

end like_terms_ratio_l260_260904


namespace blue_eyed_among_blondes_l260_260517

variable (l g b a : ℝ)

-- Given: The proportion of blondes among blue-eyed people is greater than the proportion of blondes among all people.
axiom given_condition : a / g > b / l

-- Prove: The proportion of blue-eyed people among blondes is greater than the proportion of blue-eyed people among all people.
theorem blue_eyed_among_blondes (l g b a : ℝ) (h : a / g > b / l) : a / b > g / l :=
by
  sorry

end blue_eyed_among_blondes_l260_260517


namespace probability_point_A_on_hyperbola_l260_260061

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l260_260061


namespace find_g1_l260_260074

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 1 / 2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x + 1

theorem find_g1 : g 1 = 39 / 11 :=
by
  sorry

end find_g1_l260_260074


namespace last_two_digits_10_93_10_31_plus_3_eq_08_l260_260798

def last_two_digits_fraction_floor (n m d : ℕ) : ℕ :=
  let x := 10^n
  let y := 10^m + d
  (x / y) % 100

theorem last_two_digits_10_93_10_31_plus_3_eq_08 :
  last_two_digits_fraction_floor 93 31 3 = 08 :=
by
  sorry

end last_two_digits_10_93_10_31_plus_3_eq_08_l260_260798


namespace max_arithmetic_sequence_terms_l260_260239

theorem max_arithmetic_sequence_terms
  (n : ℕ)
  (a1 : ℝ)
  (d : ℝ) 
  (sum_sq_term_cond : (a1 + (n - 1) * d / 2)^2 + (n - 1) * (a1 + d * (n - 1) / 2) ≤ 100)
  (common_diff : d = 4)
  : n ≤ 8 := 
sorry

end max_arithmetic_sequence_terms_l260_260239


namespace inhabitable_fraction_l260_260572

theorem inhabitable_fraction 
  (total_land_fraction : ℚ)
  (inhabitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1 / 3)
  (h2 : inhabitable_land_fraction = 3 / 4):
  total_land_fraction * inhabitable_land_fraction = 1 / 4 := 
by
  sorry

end inhabitable_fraction_l260_260572


namespace trapezoid_length_XY_l260_260760

noncomputable def trapezoid_problem (PQ QR RS PS : ℝ) (angle_P angle_S : ℝ) (mid_X mid_Y : ℝ) : Prop :=
  PQ ≠ 0 ∧ QR = 1500 ∧ PS = 3000 ∧ angle_P = 37 ∧ angle_S = 53 ∧
  let X := QR / 2 in
  let Y := PS / 2 in
  XY = Y - X ∧ XY = 750

theorem trapezoid_length_XY :
  trapezoid_problem PQ QR RS PS 37 53 mid_X mid_Y :=
by
  sorry

end trapezoid_length_XY_l260_260760


namespace three_digit_square_ends_with_self_l260_260871

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l260_260871


namespace inscribed_angle_sum_l260_260201

theorem inscribed_angle_sum : 
  let arcs := 24 
  let arc_to_angle (n : ℕ) := 360 / arcs * n / 2 
  (arc_to_angle 4 + arc_to_angle 6 = 75) :=
by
  sorry

end inscribed_angle_sum_l260_260201


namespace knights_count_in_meeting_l260_260323

theorem knights_count_in_meeting :
  ∃ knights, knights = 23 ∧ ∀ n : ℕ, n < 65 →
    (n < 20 → ∃ liar, liar → (liar.says (liar.previousTrueStatements - liar.previousFalseStatements = 20)))
    ∧ (n = 20 → ∃ knight, knight → (knight.says (knight.previousTrueStatements = 0 ∧ knight.previousFalseStatements = 20)))
    ∧ (20 < n → ∃ inhab, inhab (inhab.number = n) → ((inhab.isKnight = if n % 2 = 1 then true else false))) :=
sorry

end knights_count_in_meeting_l260_260323


namespace smallest_n_l260_260656

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260656


namespace find_three_digit_numbers_l260_260863

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l260_260863


namespace problem_a_problem_b_l260_260045

-- Part (a)
theorem problem_a (n: Nat) : ∃ k: ℤ, (32^ (3 * n) - 1312^ n) = 1966 * k := sorry

-- Part (b)
theorem problem_b (n: Nat) : ∃ m: ℤ, (843^ (2 * n + 1) - 1099^ (2 * n + 1) + 16^ (4 * n + 2)) = 1967 * m := sorry

end problem_a_problem_b_l260_260045


namespace max_value_expression_l260_260331

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end max_value_expression_l260_260331


namespace x_varies_as_nth_power_of_z_l260_260909

theorem x_varies_as_nth_power_of_z 
  (k j z : ℝ) 
  (h1 : ∃ y : ℝ, x = k * y^4 ∧ y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := 
 sorry

end x_varies_as_nth_power_of_z_l260_260909


namespace find_percentage_l260_260571

theorem find_percentage (x : ℝ) (h1 : x = 780) (h2 : ∀ P : ℝ, P / 100 * x = 225 - 30) : P = 25 :=
by
  -- Definitions and conditions here
  -- Recall: x = 780 and P / 100 * x = 195
  sorry

end find_percentage_l260_260571


namespace books_per_bookshelf_l260_260397

theorem books_per_bookshelf (total_bookshelves total_books books_per_bookshelf : ℕ)
  (h1 : total_bookshelves = 23)
  (h2 : total_books = 621)
  (h3 : total_books = total_bookshelves * books_per_bookshelf) :
  books_per_bookshelf = 27 :=
by 
  -- Proof goes here
  sorry

end books_per_bookshelf_l260_260397


namespace remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l260_260392

-- Definitions of angle types
def obtuse_angle (θ : ℝ) := θ > 90 ∧ θ < 180
def right_angle (θ : ℝ) := θ = 90
def acute_angle (θ : ℝ) := θ > 0 ∧ θ < 90
def straight_angle (θ : ℝ) := θ = 180

-- Proposition 1: Remaining angle when an obtuse angle is cut by a right angle is acute
theorem remaining_angle_obtuse_cut_by_right_is_acute (θ : ℝ) (φ : ℝ) 
    (h1 : obtuse_angle θ) (h2 : right_angle φ) : acute_angle (θ - φ) :=
  sorry

-- Proposition 2: Remaining angle when a straight angle is cut by an acute angle is obtuse
theorem remaining_angle_straight_cut_by_acute_is_obtuse (α : ℝ) (β : ℝ) 
    (h1 : straight_angle α) (h2 : acute_angle β) : obtuse_angle (α - β) :=
  sorry

end remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l260_260392


namespace problem_statement_l260_260859

/-- A predicate that checks if the numbers from 1 to 2n can be split into two groups 
    such that the sum of the product of the elements of each group is divisible by 2n - 1. -/
def valid_split (n : ℕ) : Prop :=
  ∃ (a b : Finset ℕ), 
  a ∪ b = Finset.range (2 * n) ∧
  a ∩ b = ∅ ∧
  (2 * n) ∣ (a.prod id + b.prod id - 1)

theorem problem_statement : 
  ∀ n : ℕ, n > 0 → valid_split n ↔ (n = 1 ∨ ∃ a : ℕ, n = 2^a ∧ a ≥ 1) :=
by
  sorry

end problem_statement_l260_260859


namespace max_product_distances_l260_260428

noncomputable def ellipse_C := {p : ℝ × ℝ | ((p.1)^2) / 9 + ((p.2)^2) / 4 = 1}

def foci_F1 : ℝ × ℝ := (c, 0) -- c is a placeholder, to be defined appropriately based on ellipse definition and properties
def foci_F2 : ℝ × ℝ := (-c, 0) -- same as above

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt (((p2.1 - p1.1)^2) + ((p2.2 - p1.2)^2))

theorem max_product_distances (M : ℝ × ℝ) (hM : M ∈ ellipse_C) :
  ∃ M ∈ ellipse_C, (distance M foci_F1) * (distance M foci_F2) = 9 := 
sorry

end max_product_distances_l260_260428


namespace percentage_difference_j_p_l260_260228

theorem percentage_difference_j_p (j p t : ℝ) (h1 : j = t * 80 / 100) 
  (h2 : t = p * (100 - t) / 100) (h3 : t = 6.25) : 
  ((p - j) / p) * 100 = 25 := 
by
  sorry

end percentage_difference_j_p_l260_260228


namespace part1_l260_260556

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) ↔ a ≤ -2 := by
  sorry

end part1_l260_260556


namespace find_integer_cube_sum_l260_260878

-- Define the problem in Lean
theorem find_integer_cube_sum : ∃ n : ℤ, n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  use 6
  sorry

end find_integer_cube_sum_l260_260878


namespace prove_math_problem_l260_260723

noncomputable def math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : Prop :=
  (x + y = 1) ∧ (x^5 + y^5 = 11)

theorem prove_math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : math_problem x y h1 h2 h3 :=
  sorry

end prove_math_problem_l260_260723


namespace smallest_positive_n_l260_260644

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260644


namespace total_money_is_102_l260_260263

-- Defining the amounts of money each person has
def Jack_money : ℕ := 26
def Ben_money : ℕ := Jack_money - 9
def Eric_money : ℕ := Ben_money - 10
def Anna_money : ℕ := Jack_money * 2

-- Defining the total amount of money
def total_money : ℕ := Eric_money + Ben_money + Jack_money + Anna_money

-- Proving the total money is 102
theorem total_money_is_102 : total_money = 102 :=
by
  -- this is where the proof would go
  sorry

end total_money_is_102_l260_260263


namespace complete_square_sum_l260_260911

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l260_260911


namespace smallest_n_satisfies_conditions_l260_260627

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260627


namespace solution_set_inequality_l260_260209

theorem solution_set_inequality (x : ℝ) : 
  (x + 5) * (3 - 2 * x) ≤ 6 ↔ (x ≤ -9/2 ∨ x ≥ 1) :=
by
  sorry  -- proof skipped as instructed

end solution_set_inequality_l260_260209


namespace roots_cubic_roots_sum_of_squares_l260_260150

variables {R : Type*} [CommRing R] {p q r s t : R}

theorem roots_cubic_roots_sum_of_squares (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
sorry

end roots_cubic_roots_sum_of_squares_l260_260150


namespace probability_of_two_black_balls_is_one_fifth_l260_260673

noncomputable def probability_of_two_black_balls (W B : Nat) : ℚ :=
  let total_balls := W + B
  let prob_black1 := (B : ℚ) / total_balls
  let prob_black2_given_black1 := (B - 1 : ℚ) / (total_balls - 1)
  prob_black1 * prob_black2_given_black1

theorem probability_of_two_black_balls_is_one_fifth : 
  probability_of_two_black_balls 8 7 = 1 / 5 := 
by
  sorry

end probability_of_two_black_balls_is_one_fifth_l260_260673


namespace gcd_equation_solution_l260_260420

theorem gcd_equation_solution (x y : ℕ) (h : Nat.gcd x y + x * y / Nat.gcd x y = x + y) : y ∣ x ∨ x ∣ y :=
 by
 sorry

end gcd_equation_solution_l260_260420


namespace find_a_l260_260166

theorem find_a (a : ℝ) : 3 * a + 150 = 360 → a = 70 := 
by 
  intro h
  sorry

end find_a_l260_260166


namespace star_4_3_l260_260303

def star (a b : ℕ) : ℕ := a^2 + a * b - b^3

theorem star_4_3 : star 4 3 = 1 := 
by
  -- sorry is used to skip the proof
  sorry

end star_4_3_l260_260303


namespace volume_of_pyramid_l260_260847

-- Define conditions
variables (x h : ℝ)
axiom x_pos : x > 0
axiom h_pos : h > 0

-- Define the main theorem/problem statement
theorem volume_of_pyramid (x h : ℝ) (x_pos : x > 0) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (1 / 6) * x^2 * h :=
by sorry

end volume_of_pyramid_l260_260847


namespace inequality_square_l260_260889

theorem inequality_square (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end inequality_square_l260_260889


namespace flux_through_section_l260_260845

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (y * z, x * z, x * y)
noncomputable def plane (x y : ℝ) : ℝ := 1 - x - y

/-- The region of interest is the section of the plane in the first octant (x, y, z ≥ 0) -/
def region_of_interest (x y : ℝ) : Prop := (0 ≤ x) ∧ (0 ≤ y) ∧ (x + y ≤ 1)

/-- The normal vector to the plane x + y + z = 1 has coordinates (1, 1, 1). -/
def normal_vector : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def unit_normal_vector (nv : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let magnitude := Real.sqrt (nv.1^2 + nv.2^2 + nv.3^2)
in (nv.1 / magnitude, nv.2 / magnitude, nv.3 / magnitude)

noncomputable def flux_integral : ℝ :=
  ∫⁻ (x : ℝ) in Icc 0 1, ∫⁻ (y : ℝ) in Icc 0 (1 - x), 
    (vector_field x y (plane x y)) (unit_normal_vector normal_vector) (x, y)

/-- The flux of the vector field through the section of the plane in the first octant, 
along the normal vector of the plane, is 1/12. -/
theorem flux_through_section : flux_integral = 1 / 12 := by
  sorry

end flux_through_section_l260_260845


namespace range_of_a_l260_260851

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the main theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, tensor (x - a) (x + a) < 1) → 
  (-((1 : ℝ) / 2) < a ∧ a < (3 : ℝ) / 2) :=
by
  sorry

end range_of_a_l260_260851


namespace vectors_collinear_has_solution_l260_260294

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x^2 - 1, 2 + x)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinearity condition (cross product must be zero) as a function
def collinear (x : ℝ) : Prop := (a x).1 * (b x).2 - (b x).1 * (a x).2 = 0

-- The proof statement
theorem vectors_collinear_has_solution (x : ℝ) (h : collinear x) : x = -1 / 2 :=
sorry

end vectors_collinear_has_solution_l260_260294


namespace blue_first_yellow_second_probability_l260_260248

open Classical

-- Definition of initial conditions
def total_marbles : Nat := 3 + 4 + 9
def blue_marbles : Nat := 3
def yellow_marbles : Nat := 4
def pink_marbles : Nat := 9

-- Probability functions
def probability_first_blue : ℚ := blue_marbles / total_marbles
def probability_second_yellow_given_blue : ℚ := yellow_marbles / (total_marbles - 1)

-- Combined probability
def combined_probability_first_blue_second_yellow : ℚ := 
  probability_first_blue * probability_second_yellow_given_blue

-- Theorem statement
theorem blue_first_yellow_second_probability :
  combined_probability_first_blue_second_yellow = 1 / 20 :=
by
  -- Proof will be provided here
  sorry

end blue_first_yellow_second_probability_l260_260248


namespace linear_function_increasing_l260_260027

variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x1 < x2)
variable (P1_eq : y1 = 2 * x1 + 1)
variable (P2_eq : y2 = 2 * x2 + 1)

theorem linear_function_increasing (hx : x1 < x2) (P1_eq : y1 = 2 * x1 + 1) (P2_eq : y2 = 2 * x2 + 1) 
    : y1 < y2 := sorry

end linear_function_increasing_l260_260027


namespace total_rocks_needed_l260_260188

def rocks_already_has : ℕ := 64
def rocks_needed : ℕ := 61

theorem total_rocks_needed : rocks_already_has + rocks_needed = 125 :=
by
  sorry

end total_rocks_needed_l260_260188


namespace tteokbokki_cost_l260_260925

theorem tteokbokki_cost (P : ℝ) (h1 : P / 2 - P * (3 / 16) = 2500) : P / 2 = 4000 :=
by
  sorry

end tteokbokki_cost_l260_260925


namespace misha_is_lying_l260_260989

theorem misha_is_lying
  (truth_tellers_scores : Fin 9 → ℕ)
  (h_all_odd : ∀ i, truth_tellers_scores i % 2 = 1)
  (total_scores_truth_tellers : (Fin 9 → ℕ) → ℕ)
  (h_sum_scores : total_scores_truth_tellers truth_tellers_scores = 18) :
  ∀ (misha_score : ℕ), misha_score = 2 → misha_score % 2 = 1 → False :=
by
  intros misha_score hms hmo
  sorry

end misha_is_lying_l260_260989


namespace distinct_equilateral_triangles_in_polygon_l260_260552

noncomputable def num_distinct_equilateral_triangles (P : Finset (Fin 10)) : Nat :=
  90

theorem distinct_equilateral_triangles_in_polygon (P : Finset (Fin 10)) :
  P.card = 10 →
  num_distinct_equilateral_triangles P = 90 :=
by
  intros
  sorry

end distinct_equilateral_triangles_in_polygon_l260_260552


namespace find_possible_values_of_a_l260_260791

noncomputable def find_a (x y a : ℝ) : Prop :=
  (x + y = a) ∧ (x^3 + y^3 = a) ∧ (x^5 + y^5 = a)

theorem find_possible_values_of_a (a : ℝ) :
  (∃ x y : ℝ, find_a x y a) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
sorry

end find_possible_values_of_a_l260_260791


namespace field_ratio_l260_260614

theorem field_ratio (w : ℝ) (h : ℝ) (pond_len : ℝ) (field_len : ℝ) 
  (h1 : pond_len = 8) 
  (h2 : field_len = 112) 
  (h3 : w > 0) 
  (h4 : field_len = w * h) 
  (h5 : pond_len * pond_len = (1 / 98) * (w * h * h)) : 
  field_len / h = 2 := 
by 
  sorry

end field_ratio_l260_260614


namespace two_roots_range_a_l260_260440

noncomputable def piecewise_func (x : ℝ) : ℝ :=
if x ≤ 1 then (1/3) * x + 1 else Real.log x

theorem two_roots_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ piecewise_func x1 = a * x1 ∧ piecewise_func x2 = a * x2) ↔ (1/3 < a ∧ a < 1/Real.exp 1) :=
sorry

end two_roots_range_a_l260_260440


namespace rachel_math_homework_pages_l260_260941

theorem rachel_math_homework_pages (M : ℕ) 
  (h1 : 23 = M + (M + 3)) : M = 10 :=
by {
  sorry
}

end rachel_math_homework_pages_l260_260941


namespace find_r_l260_260154

variable (n : ℕ) (q r : ℝ)

-- n must be a positive natural number
axiom n_pos : n > 0

-- q is a positive real number and not equal to 1
axiom q_pos : q > 0
axiom q_ne_one : q ≠ 1

-- Define the sequence sum S_n according to the problem statement
def S_n (n : ℕ) (q r : ℝ) : ℝ := q^n + r

-- The goal is to prove that the correct value of r is -1
theorem find_r : r = -1 :=
sorry

end find_r_l260_260154


namespace bus_A_speed_l260_260843

variable (v_A v_B : ℝ)
variable (h1 : v_A - v_B = 15)
variable (h2 : v_A + v_B = 75)

theorem bus_A_speed : v_A = 45 := sorry

end bus_A_speed_l260_260843


namespace input_x_for_y_16_l260_260216

noncomputable def output_y_from_input_x (x : Int) : Int :=
if x < 0 then (x + 1) * (x + 1)
else (x - 1) * (x - 1)

theorem input_x_for_y_16 (x : Int) (y : Int) (h : y = 16) :
  output_y_from_input_x x = y ↔ (x = 5 ∨ x = -5) :=
by
  sorry

end input_x_for_y_16_l260_260216


namespace value_of_s_l260_260928

theorem value_of_s (s : ℝ) : (3 * (-1)^5 + 2 * (-1)^4 - (-1)^3 + (-1)^2 - 4 * (-1) + s = 0) → (s = -5) :=
by
  intro h
  sorry

end value_of_s_l260_260928


namespace mike_spent_l260_260347

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l260_260347


namespace mans_rate_in_still_water_l260_260382

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water_l260_260382


namespace knights_count_l260_260324

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l260_260324


namespace fib_subsequence_fib_l260_260595

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fib_subsequence_fib (p : ℕ) (hp : p > 0) :
  ∀ n : ℕ, fibonacci ((n - 1) * p) + fibonacci (n * p) = fibonacci ((n + 1) * p) := 
by
  sorry

end fib_subsequence_fib_l260_260595


namespace ferry_max_weight_capacity_l260_260988

def automobile_max_weight : ℝ := 3200
def automobile_count : ℝ := 62.5
def pounds_to_tons : ℝ := 2000

theorem ferry_max_weight_capacity : 
  (automobile_max_weight * automobile_count) / pounds_to_tons = 100 := 
by 
  sorry

end ferry_max_weight_capacity_l260_260988


namespace area_of_region_is_12_l260_260958

def region_area : ℝ :=
  let f1 (x : ℝ) : ℝ := |x - 2|
  let f2 (x : ℝ) : ℝ := 5 - |x + 1|
  let valid_region (x y : ℝ) : Prop := f1 x ≤ y ∧ y ≤ f2 x
  12

theorem area_of_region_is_12 :
  ∃ (area : ℝ), region_area = 12 := by
  use 12
  sorry

end area_of_region_is_12_l260_260958


namespace jamshid_taimour_painting_problem_l260_260762

/-- Jamshid and Taimour Painting Problem -/
theorem jamshid_taimour_painting_problem (T : ℝ) (h1 : T > 0)
  (h2 : 1 / T + 2 / T = 1 / 5) : T = 15 :=
by
  -- solving the theorem
  sorry

end jamshid_taimour_painting_problem_l260_260762


namespace second_person_time_l260_260376

theorem second_person_time (x : ℝ) (h1 : ∀ t : ℝ, t = 3) 
(h2 : (1/3 + 1/x) = 5/12) : x = 12 := 
by sorry

end second_person_time_l260_260376


namespace value_of_M_l260_260447

theorem value_of_M (M : ℝ) (h : (25 / 100) * M = (35 / 100) * 1800) : M = 2520 := 
sorry

end value_of_M_l260_260447


namespace count_multiples_of_30_l260_260545

theorem count_multiples_of_30 (a b n : ℕ) (h1 : a = 900) (h2 : b = 27000) 
    (h3 : ∃ n, 30 * n = a) (h4 : ∃ n, 30 * n = b) : 
    (b - a) / 30 + 1 = 871 := 
by
    sorry

end count_multiples_of_30_l260_260545


namespace long_show_episode_duration_is_one_hour_l260_260374

-- Definitions for the given conditions
def total_shows : ℕ := 2
def short_show_length : ℕ := 24
def short_show_episode_duration : ℝ := 0.5
def long_show_episodes : ℕ := 12
def total_viewing_time : ℝ := 24

-- Definition of the length of each episode of the longer show
def long_show_episode_length (L : ℝ) : Prop :=
  (short_show_length * short_show_episode_duration) + (long_show_episodes * L) = total_viewing_time

-- Main statement to prove
theorem long_show_episode_duration_is_one_hour : long_show_episode_length 1 :=
by
  -- Proof placeholder
  sorry

end long_show_episode_duration_is_one_hour_l260_260374


namespace greatest_possible_value_of_a_l260_260796

theorem greatest_possible_value_of_a (a : ℤ) (h1 : ∃ x : ℤ, x^2 + a*x = -30) (h2 : 0 < a) :
  a ≤ 31 :=
sorry

end greatest_possible_value_of_a_l260_260796


namespace find_smallest_n_l260_260555

theorem find_smallest_n 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (h1 : a_n 1 + a_n 2 = 9 / 2)
    (h2 : S_n 4 = 45 / 8)
    (h3 : ∀ n, S_n n = (1 / 2) * n * (a_n 1 + a_n n)) :
    ∃ n : ℕ, a_n n < 1 / 10 ∧ ∀ m : ℕ, m < n → a_n m ≥ 1 / 10 := 
sorry

end find_smallest_n_l260_260555


namespace at_most_one_hired_l260_260840

-- Definition of the events and their probabilities
def P_A : ℝ := 0.5
def P_B : ℝ := 0.6

-- Assumption of independence of events A and B
def independent_events : Prop := 
  ∀ (A B : Prop), (P_A * P_B) = 0.3

-- Statement of the problem
theorem at_most_one_hired : 
  (P_A + P_B - 2 * 0.3) = 0.7 :=
by
  sorry

end at_most_one_hired_l260_260840


namespace sales_tax_difference_l260_260541

theorem sales_tax_difference :
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  difference = 0.375 :=
by
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  exact sorry

end sales_tax_difference_l260_260541


namespace knights_count_l260_260322

theorem knights_count :
  ∀ (total_inhabitants : ℕ) 
  (P : (ℕ → Prop)) 
  (H : (∀ i, i < total_inhabitants → (P i ↔ (∃ T F, T = F - 20 ∧ T = ∑ j in finset.range i, if P j then 1 else 0 ∧ F = i - T))),
  total_inhabitants = 65 →
  (∃ knights : ℕ, knights = 23) :=
begin
  intros total_inhabitants P H inj_id,
  sorry  -- proof goes here
end

end knights_count_l260_260322


namespace balls_total_correct_l260_260315

-- Definitions based on the problem conditions
def red_balls_initial : ℕ := 16
def blue_balls : ℕ := 2 * red_balls_initial
def red_balls_lost : ℕ := 6
def red_balls_remaining : ℕ := red_balls_initial - red_balls_lost
def total_balls_after : ℕ := 74
def nonblue_red_balls_remaining : ℕ := red_balls_remaining + blue_balls

-- Goal: Find the number of yellow balls
def yellow_balls_bought : ℕ := total_balls_after - nonblue_red_balls_remaining

theorem balls_total_correct :
  yellow_balls_bought = 32 :=
by
  -- Proof would go here
  sorry

end balls_total_correct_l260_260315


namespace probability_three_digit_divisible_by_3_l260_260887

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def remainder_groups : (Finset ℕ) × (Finset ℕ) × (Finset ℕ) :=
  ({1, 4, 7}, {2, 5, 8}, {0, 3, 6, 9})

noncomputable def total_combinations :=
  fintype.card (digits.erase 0).to_finset * fintype.card (digits.erase 0).to_finset * fintype.card (digits.erase 1).to_finset

noncomputable def favorable_combinations : ℕ :=
  -- placeholder calculation for valid combinations, assume we have 228 valid cases.
  228

noncomputable def probability_divisible_by_3 :=
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem probability_three_digit_divisible_by_3 :
  probability_divisible_by_3 = 19 / 54 :=
sorry

end probability_three_digit_divisible_by_3_l260_260887


namespace ten_thousand_points_length_l260_260189

theorem ten_thousand_points_length (a b : ℝ) (d : ℝ) 
  (h1 : d = a / 99) 
  (h2 : b = 9999 * d) : b = 101 * a := by
  sorry

end ten_thousand_points_length_l260_260189


namespace value_of_a_l260_260451

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) → (a = 2 ∨ a = 3 ∨ a = 4) :=
by sorry

end value_of_a_l260_260451


namespace quadratic_rewrite_ab_l260_260778

theorem quadratic_rewrite_ab : 
  ∃ (a b c : ℤ), (16*(x:ℝ)^2 - 40*x + 24 = (a*x + b)^2 + c) ∧ (a * b = -20) :=
by {
  sorry
}

end quadratic_rewrite_ab_l260_260778


namespace tracy_initial_candies_l260_260375

theorem tracy_initial_candies (x : ℕ) (consumed_candies : ℕ) (remaining_candies_given_rachel : ℕ) (remaining_candies_given_monica : ℕ) (candies_eaten_by_tracy : ℕ) (candies_eaten_by_mom : ℕ) 
  (brother_candies_taken : ℕ) (final_candies : ℕ) (h_consume : consumed_candies = 2 / 5 * x) (h_remaining1 : remaining_candies_given_rachel = 1 / 3 * (3 / 5 * x)) 
  (h_remaining2 : remaining_candies_given_monica = 1 / 6 * (3 / 5 * x)) (h_left_after_friends : 3 / 5 * x - (remaining_candies_given_rachel + remaining_candies_given_monica) = 3 / 10 * x)
  (h_candies_left : 3 / 10 * x - (candies_eaten_by_tracy + candies_eaten_by_mom) = final_candies + brother_candies_taken) (h_eaten_tracy : candies_eaten_by_tracy = 10)
  (h_eaten_mom : candies_eaten_by_mom = 10) (h_final : final_candies = 6) (h_brother_bound : 2 ≤ brother_candies_taken ∧ brother_candies_taken ≤ 6) : x = 100 := 
by 
  sorry

end tracy_initial_candies_l260_260375


namespace cos_double_angle_l260_260148

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) : Real.cos (2 * α) = -1 / 3 := 
  sorry

end cos_double_angle_l260_260148


namespace smallest_positive_n_l260_260643

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260643


namespace find_n_for_divisibility_l260_260910

def digit_sum_odd_positions := 8 + 4 + 5 + 6 -- The sum of the digits in odd positions
def digit_sum_even_positions (n : ℕ) := 5 + n + 2 -- The sum of the digits in even positions

def is_divisible_by_11 (n : ℕ) := (digit_sum_odd_positions - digit_sum_even_positions n) % 11 = 0

theorem find_n_for_divisibility : is_divisible_by_11 5 :=
by
  -- Proof would go here (but according to the instructions, we'll insert a placeholder)
  sorry

end find_n_for_divisibility_l260_260910


namespace sequence_formula_l260_260157

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
  (a 1 = 11) ∧
  (a 2 = 102) ∧
  (a 3 = 1003) ∧
  (a 4 = 10004) →
  ∀ n, a n = 10^n + n := by
  sorry

end sequence_formula_l260_260157


namespace monotonicity_of_f_solve_inequality_l260_260179

noncomputable def f (x : ℝ) : ℝ := sorry

def f_defined : ∀ x > 0, ∃ y, f y = f x := sorry

axiom functional_eq : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y 

axiom f_gt_zero : ∀ x, x > 1 → f x > 0

theorem monotonicity_of_f : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality (x : ℝ) (h1 : f 2 = 1) (h2 : 0 < x) : 
  f x + f (x - 3) ≤ 2 ↔ 3 < x ∧ x ≤ 4 :=
sorry

end monotonicity_of_f_solve_inequality_l260_260179


namespace find_x_in_sequence_l260_260030

theorem find_x_in_sequence :
  ∃ x y z : Int, (z + 3 = 5) ∧ (y + z = 5) ∧ (x + y = 2) ∧ (x = -1) :=
by
  use -1, 3, 2
  sorry

end find_x_in_sequence_l260_260030


namespace at_least_one_ge_one_l260_260585

theorem at_least_one_ge_one (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 2) : 
  max (|a - b|) (max (|b - c|) (|c - a|)) ≥ 1 :=
by 
  sorry

end at_least_one_ge_one_l260_260585


namespace cube_volume_l260_260516

theorem cube_volume (A : ℝ) (hA : A = 96) (s : ℝ) (hS : A = 6 * s^2) : s^3 = 64 := by
  sorry

end cube_volume_l260_260516


namespace total_students_l260_260212

theorem total_students : 
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  in total = 86 :=
by
  let grade_3 := 19
  let grade_4 := 2 * grade_3
  let boys_2 := 10
  let girls_2 := 19
  let grade_2 := boys_2 + girls_2
  let total := grade_2 + grade_3 + grade_4
  sorry

end total_students_l260_260212


namespace find_smallest_n_l260_260637

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260637


namespace evaluate_f_5_minus_f_neg_5_l260_260745

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l260_260745


namespace animal_counts_l260_260069

-- Definitions based on given conditions
def ReptileHouse (R : ℕ) : ℕ := 3 * R - 5
def Aquarium (ReptileHouse : ℕ) : ℕ := 2 * ReptileHouse
def Aviary (Aquarium RainForest : ℕ) : ℕ := (Aquarium - RainForest) + 3

-- The main theorem statement
theorem animal_counts
  (R : ℕ)
  (ReptileHouse_eq : ReptileHouse R = 16)
  (A : ℕ := Aquarium 16)
  (V : ℕ := Aviary A R) :
  (R = 7) ∧ (A = 32) ∧ (V = 28) :=
by
  sorry

end animal_counts_l260_260069


namespace part_one_part_two_l260_260337

noncomputable def f (a x : ℝ) : ℝ :=
  |x + (1 / a)| + |x - a + 1|

theorem part_one (a : ℝ) (h : a > 0) (x : ℝ) : f a x ≥ 1 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : f a 3 < 11 / 2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part_one_part_two_l260_260337


namespace isosceles_right_triangle_leg_length_l260_260612

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l260_260612


namespace inequality_proof_l260_260174

theorem inequality_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end inequality_proof_l260_260174


namespace smallest_n_satisfies_conditions_l260_260624

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260624


namespace evaluate_expression_l260_260707

theorem evaluate_expression :
  (3025^2 : ℝ) / ((305^2 : ℝ) - (295^2 : ℝ)) = 1525.10417 :=
by
  sorry

end evaluate_expression_l260_260707


namespace sequence_general_term_l260_260726

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n, n ≥ 1 → a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end sequence_general_term_l260_260726


namespace rational_solution_cos_eq_l260_260788

theorem rational_solution_cos_eq {q : ℚ} (h0 : 0 < q) (h1 : q < 1) (heq : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2 / 3 := 
sorry

end rational_solution_cos_eq_l260_260788


namespace not_possible_100_odd_sequence_l260_260032

def is_square_mod_8 (n : ℤ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 1 ∨ n % 8 = 4

def sum_consecutive_is_square_mod_8 (seq : List ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i + k ≤ seq.length →
  is_square_mod_8 (seq.drop i |>.take k |>.sum)

def valid_odd_sequence (seq : List ℤ) : Prop :=
  seq.length = 100 ∧
  (∀ n ∈ seq, n % 2 = 1) ∧
  sum_consecutive_is_square_mod_8 seq 5 ∧
  sum_consecutive_is_square_mod_8 seq 9

theorem not_possible_100_odd_sequence :
  ¬∃ seq : List ℤ, valid_odd_sequence seq :=
by
  sorry

end not_possible_100_odd_sequence_l260_260032


namespace min_value_of_M_l260_260929

noncomputable def f (p q x : ℝ) : ℝ := x^2 + p * x + q

theorem min_value_of_M (p q M : ℝ) :
  (M = max (|f p q 1|) (max (|f p q (-1)|) (|f p q 0|))) →
  (0 > f p q 1 → 0 > f p q (-1) → 0 > f p q 0 → M = 1 / 2) :=
sorry

end min_value_of_M_l260_260929


namespace measure_of_B_l260_260754

-- Define the conditions (angles and their relationships)
variable (angle_P angle_R angle_O angle_B angle_L angle_S : ℝ)
variable (sum_of_angles : angle_P + angle_R + angle_O + angle_B + angle_L + angle_S = 720)
variable (supplementary_O_S : angle_O + angle_S = 180)
variable (right_angle_L : angle_L = 90)
variable (congruent_angles : angle_P = angle_R ∧ angle_R = angle_B)

-- Prove the measure of angle B
theorem measure_of_B : angle_B = 150 := by
  sorry

end measure_of_B_l260_260754


namespace largest_angle_in_triangle_l260_260210

theorem largest_angle_in_triangle (a b c : ℝ)
  (h1 : a + b = (4 / 3) * 90)
  (h2 : b = a + 36)
  (h3 : a + b + c = 180) :
  max a (max b c) = 78 :=
sorry

end largest_angle_in_triangle_l260_260210


namespace max_plus_min_value_of_y_eq_neg4_l260_260290

noncomputable def y (x : ℝ) : ℝ := (2 * (Real.sin x) ^ 2 + Real.sin (3 * x / 2) - 4) / ((Real.sin x) ^ 2 + 2 * (Real.cos x) ^ 2)

theorem max_plus_min_value_of_y_eq_neg4 (M m : ℝ) (hM : ∃ x : ℝ, y x = M) (hm : ∃ x : ℝ, y x = m) :
  M + m = -4 := sorry

end max_plus_min_value_of_y_eq_neg4_l260_260290


namespace problem_180_180_minus_12_l260_260093

namespace MathProof

theorem problem_180_180_minus_12 :
  180 * (180 - 12) - (180 * 180 - 12) = -2148 := 
by
  -- Placeholders for computation steps
  sorry

end MathProof

end problem_180_180_minus_12_l260_260093


namespace paperboy_problem_l260_260243

noncomputable def delivery_ways (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else if n = 2 then 4
  else if n = 3 then 8
  else if n = 4 then 15
  else delivery_ways (n - 1) + delivery_ways (n - 2) + delivery_ways (n - 3) + delivery_ways (n - 4)

theorem paperboy_problem : delivery_ways 12 = 2872 :=
  sorry

end paperboy_problem_l260_260243


namespace avg_growth_rate_production_lines_target_infeasibility_of_60000_l260_260974

-- 1. Prove that the average growth rate of sales in the first three quarters is 20%
theorem avg_growth_rate (q1_sales q3_sales : ℕ) (x : ℚ) :
  q1_sales = 20000 → q3_sales = 28800 → (1 + x) ^ 2 = q3_sales / q1_sales → x = 0.2 :=
by
  intros
  sorry

-- 2. Prove that 5 production lines are required to achieve a production target of 26,000 units per quarter
theorem production_lines_target (max_capacity per_reduction : ℕ) (target_production : ℕ) :
  max_capacity = 6000 → per_reduction = 200 → target_production = 26000 →
  ∃ m : ℕ, 30 * m - m^2 = 100 ∧ 26 ≤ (1 + m) * (6000 - 200 * m) := 
by
  intros
  sorry

-- 3. Prove that it is not possible to produce 60,000 units per quarter with any number of production lines
theorem infeasibility_of_60000 (max_capacity per_reduction : ℕ) (target_production : ℕ) :
  max_capacity = 6000 → per_reduction = 200 → target_production = 60000 →
  ¬ ∃ n : ℕ, n^2 - 30 * n + 270 = 0 ∧ 60 ≤ (1 + n) * (6000 - 200 * n) :=
by
  intros
  sorry

end avg_growth_rate_production_lines_target_infeasibility_of_60000_l260_260974


namespace total_children_on_bus_after_stop_l260_260387

theorem total_children_on_bus_after_stop (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 18) (h2 : additional = 7) : total = 25 :=
by sorry

end total_children_on_bus_after_stop_l260_260387


namespace sufficient_but_not_necessary_condition_l260_260276

theorem sufficient_but_not_necessary_condition 
  (a : ℕ → ℤ) 
  (h : ∀ n, |a (n + 1)| < a n) : 
  (∀ n, a (n + 1) < a n) ∧ 
  ¬(∀ n, a (n + 1) < a n → |a (n + 1)| < a n) := 
by 
  sorry

end sufficient_but_not_necessary_condition_l260_260276


namespace find_a_l260_260714

variable (a x y : ℝ)

theorem find_a (h1 : x / (2 * y) = 3 / 2) (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) : a = 7 :=
sorry

end find_a_l260_260714


namespace halfway_between_one_eighth_and_one_tenth_l260_260799

theorem halfway_between_one_eighth_and_one_tenth :
  (1 / 8 + 1 / 10) / 2 = 9 / 80 :=
by
  sorry

end halfway_between_one_eighth_and_one_tenth_l260_260799


namespace smallest_n_satisfies_conditions_l260_260628

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260628


namespace prism_base_shape_l260_260996

theorem prism_base_shape (n : ℕ) (hn : 3 * n = 12) : n = 4 := by
  sorry

end prism_base_shape_l260_260996


namespace g_s_difference_l260_260748

def g (n : ℤ) : ℤ := n^3 + 3 * n^2 + 3 * n + 1

theorem g_s_difference (s : ℤ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end g_s_difference_l260_260748


namespace problem_solution_l260_260766

noncomputable def P (x y : ℕ) : ℕ := ...

theorem problem_solution :
  (∃ P : ℕ → ℕ → ℕ, 
    (∀ x y, P x y = ∑ i in Finset.range (x + y + 1), Nat.choose (x + y) i) ∧
    (∀ x, y, P x y = Finset.sum (Finset.range (x + y + 1)) (λ i, Nat.choose (x + y) i)) ∧
    (∀ x, y, P x y = ∏ i in Finset.range x, Nat.choose (i + y + x - i) x) ∧
    ∀ x, y, P(x,y).degree_x ≤ 2020 ∧ P(x,y).degree_y ≤ 2020 ) →
  P 4040 4040 % 2017 = 1555 :=
by mathlib_example sorry

end problem_solution_l260_260766


namespace frog_lands_on_corner_once_in_four_hops_l260_260990

/-- The simplified grid for the frog hopping problem -/
inductive GridPos
| Center
| Edge
| Corner

/-- Define initial conditions and movement properties -/
def is_wraparound (p : GridPos) : Prop := sorry -- to define wraparound movement

/-- Define hopping probability based on the current position -/
def hop_prob (from to: GridPos) : ℚ :=
match from, to with
| GridPos.Center, GridPos.Edge   => 1
| GridPos.Edge, GridPos.Edge   => 1 / 2
| GridPos.Edge, GridPos.Corner => 1 / 4
| GridPos.Edge, GridPos.Center => 1 / 4
| _, _                           => 0 -- Invalid transitions within four hops
end

/-- Starting from the center with up to four hops, the frog lands exactly once 
  on a corner with probability 25/32 -/
theorem frog_lands_on_corner_once_in_four_hops :
  (Prob (λ paths, paths.start = GridPos.Center → ∃ hops ≤ 4, paths(hops) = GridPos.Corner)) = 25 / 32 :=
by
  sorry

end frog_lands_on_corner_once_in_four_hops_l260_260990


namespace total_amount_spent_l260_260016

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l260_260016


namespace part1_part2_l260_260325

-- Definitions for the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Proof statement for the first part
theorem part1 (m : ℝ) (h : m = 4) : A ∪ B m = { x | -2 ≤ x ∧ x ≤ 7 } :=
sorry

-- Proof statement for the second part
theorem part2 (h : ∀ {m : ℝ}, B m ⊆ A) : ∀ m : ℝ, m ∈ Set.Iic 3 :=
sorry

end part1_part2_l260_260325


namespace complete_square_sum_l260_260912

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l260_260912


namespace find_a_cubed_l260_260482

-- Definitions based on conditions
def varies_inversely (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement with given conditions
theorem find_a_cubed (a b : ℝ) (k : ℝ) (h1 : varies_inversely a b)
    (h2 : a = 2) (h3 : b = 4) (k_val : k = 2048) (b_new : b = 8) : a^3 = 1 / 2 :=
sorry

end find_a_cubed_l260_260482


namespace evaluate_f_5_minus_f_neg_5_l260_260744

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l260_260744


namespace probability_point_on_hyperbola_l260_260062

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l260_260062


namespace gum_distribution_l260_260036

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l260_260036


namespace imaginary_unit_sum_l260_260449

-- Define that i is the imaginary unit, which satisfies \(i^2 = -1\)
def is_imaginary_unit (i : ℂ) := i^2 = -1

-- The theorem to be proven: i + i^2 + i^3 + i^4 = 0 given that i is the imaginary unit
theorem imaginary_unit_sum (i : ℂ) (h : is_imaginary_unit i) : 
  i + i^2 + i^3 + i^4 = 0 := 
sorry

end imaginary_unit_sum_l260_260449


namespace folded_paper_perimeter_l260_260831

theorem folded_paper_perimeter (L W : ℝ) 
  (h1 : 2 * L + W = 34)         -- Condition 1
  (h2 : L * W = 140)            -- Condition 2
  : 2 * W + L = 38 :=           -- Goal
sorry

end folded_paper_perimeter_l260_260831


namespace probability_point_on_hyperbola_l260_260059

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l260_260059


namespace parallelogram_construction_l260_260280

theorem parallelogram_construction 
  (α : ℝ) (hα : 0 ≤ α ∧ α < 180)
  (A B : (ℝ × ℝ))
  (in_angle : (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ α ∧ 
               ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ α))
  (C D : (ℝ × ℝ)) :
  ∃ O : (ℝ × ℝ), 
    O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) :=
sorry

end parallelogram_construction_l260_260280


namespace square_root_ratio_area_l260_260602

theorem square_root_ratio_area (side_length_C side_length_D : ℕ) (hC : side_length_C = 45) (hD : side_length_D = 60) : 
  Real.sqrt ((side_length_C^2 : ℝ) / (side_length_D^2 : ℝ)) = 3 / 4 :=
by
  rw [hC, hD]
  sorry

end square_root_ratio_area_l260_260602


namespace ratio_of_boys_to_total_l260_260051

theorem ratio_of_boys_to_total (p_b p_g : ℝ) (h1 : p_b + p_g = 1) (h2 : p_b = (2 / 3) * p_g) :
  p_b = 2 / 5 :=
by
  sorry

end ratio_of_boys_to_total_l260_260051


namespace four_people_fill_pool_together_in_12_minutes_l260_260318

def combined_pool_time (j s t e : ℕ) : ℕ := 
  1 / ((1 / j) + (1 / s) + (1 / t) + (1 / e))

theorem four_people_fill_pool_together_in_12_minutes : 
  ∀ (j s t e : ℕ), j = 30 → s = 45 → t = 90 → e = 60 → combined_pool_time j s t e = 12 := 
by 
  intros j s t e h_j h_s h_t h_e
  unfold combined_pool_time
  rw [h_j, h_s, h_t, h_e]
  have r1 : 1 / 30 = 1 / 30 := rfl
  have r2 : 1 / 45 = 1 / 45 := rfl
  have r3 : 1 / 90 = 1 / 90 := rfl
  have r4 : 1 / 60 = 1 / 60 := rfl
  rw [r1, r2, r3, r4]
  norm_num
  sorry

end four_people_fill_pool_together_in_12_minutes_l260_260318


namespace smallest_n_45_l260_260632

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260632


namespace max_f_alpha_side_a_l260_260737

noncomputable def a_vec (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)
noncomputable def b_vec (α : ℝ) : ℝ × ℝ := (6 * Real.sin α + Real.cos α, 7 * Real.sin α - 2 * Real.cos α)

noncomputable def f (α : ℝ) : ℝ := (a_vec α).1 * (b_vec α).1 + (a_vec α).2 * (b_vec α).2

theorem max_f_alpha : ∀ α : ℝ, f α ≤ 4 * Real.sqrt 2 + 2 :=
by
sorry

theorem side_a (A : ℝ) (b c : ℝ) (h1 : f A = 6) (h2 : 1/2 * b * c * Real.sin A = 3) (h3 : b + c = 2 + 3 * Real.sqrt 2) : 
  ∃ a : ℝ, a = Real.sqrt 10 :=
by
sorry

end max_f_alpha_side_a_l260_260737


namespace largest_5_digit_integer_congruent_to_19_mod_26_l260_260622

theorem largest_5_digit_integer_congruent_to_19_mod_26 :
  ∃ n : ℕ, 10000 ≤ 26 * n + 19 ∧ 26 * n + 19 < 100000 ∧ (26 * n + 19 ≡ 19 [MOD 26]) ∧ 26 * n + 19 = 99989 :=
by
  sorry

end largest_5_digit_integer_congruent_to_19_mod_26_l260_260622


namespace pythagorean_theorem_l260_260052

theorem pythagorean_theorem (a b c : ℕ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l260_260052


namespace triangle_sets_l260_260662

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end triangle_sets_l260_260662


namespace find_int_k_l260_260221

theorem find_int_k (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^3) :
  K = 11 :=
by
  sorry

end find_int_k_l260_260221


namespace locus_of_Y_right_angled_triangle_l260_260893

-- Conditions definitions
variables {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (b c m : ℝ) -- Coordinates and slopes related to the problem
variables (x : ℝ) -- Independent variable for the locus line

-- The problem statement
theorem locus_of_Y_right_angled_triangle 
  (A_right_angle : ∀ (α β : ℝ), α * β = 0) 
  (perpendicular_lines : b ≠ m * c) 
  (no_coincide : (b^2 * m - 2 * b * c - c^2 * m) ≠ 0) :
  ∃ (y : ℝ), y = (2 * b * c * (b * m - c) - x * (b^2 + 2 * b * c * m - c^2)) / (b^2 * m - 2 * b * c - c^2 * m) := 
sorry

end locus_of_Y_right_angled_triangle_l260_260893


namespace Jill_arrives_30_minutes_before_Jack_l260_260317

theorem Jill_arrives_30_minutes_before_Jack
  (d : ℝ) (v_J : ℝ) (v_K : ℝ)
  (h₀ : d = 3) (h₁ : v_J = 12) (h₂ : v_K = 4) :
  (d / v_K - d / v_J) * 60 = 30 :=
by
  sorry

end Jill_arrives_30_minutes_before_Jack_l260_260317


namespace probability_one_first_class_product_l260_260092

-- Define the probabilities for the interns processing first-class products
def P_first_intern_first_class : ℚ := 2 / 3
def P_second_intern_first_class : ℚ := 3 / 4

-- Define the events 
def P_A1 : ℚ := P_first_intern_first_class * (1 - P_second_intern_first_class)
def P_A2 : ℚ := (1 - P_first_intern_first_class) * P_second_intern_first_class

-- Probability of exactly one of the two parts being first-class product
def P_one_first_class_product : ℚ := P_A1 + P_A2

-- Theorem to be proven: the probability is 5/12
theorem probability_one_first_class_product : 
    P_one_first_class_product = 5 / 12 :=
by
  -- Proof goes here
  sorry

end probability_one_first_class_product_l260_260092


namespace number_of_ways_to_distribute_balls_l260_260565

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l260_260565


namespace sum_of_ages_is_29_l260_260494

theorem sum_of_ages_is_29 (age1 age2 age3 : ℕ) (h1 : age1 = 9) (h2 : age2 = 9) (h3 : age3 = 11) :
  age1 + age2 + age3 = 29 := by
  -- skipping the proof
  sorry

end sum_of_ages_is_29_l260_260494


namespace max_min_XY_XZ_diff_zero_l260_260581

theorem max_min_XY_XZ_diff_zero (YZ : ℝ) (XM : ℝ) (M : ℝ → ℝ) (x : ℝ) (h : ℝ):
  (YZ = 10) →
  (XM = 6) →
  (∀ y z : ℝ, M y = M z) →
  ∃ N n : ℝ, N = n ∧ N - n = 0 :=
by
  intro YZ_eq XM_eq M_midpoint
  use 122
  use 122
  split
  · sorry -- Proof that N = n = 122
  · sorry -- Proof that N - n = 0

end max_min_XY_XZ_diff_zero_l260_260581


namespace smallest_n_l260_260654

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l260_260654


namespace polygon_area_leq_17_point_5_l260_260489

theorem polygon_area_leq_17_point_5 (proj_OX proj_bisector_13 proj_OY proj_bisector_24 : ℝ)
  (h1: proj_OX = 4)
  (h2: proj_bisector_13 = 3 * Real.sqrt 2)
  (h3: proj_OY = 5)
  (h4: proj_bisector_24 = 4 * Real.sqrt 2)
  (S : ℝ) :
  S ≤ 17.5 := sorry

end polygon_area_leq_17_point_5_l260_260489


namespace remainder_of_sum_l260_260094

theorem remainder_of_sum (a b c : ℕ) (h1 : a % 15 = 8) (h2 : b % 15 = 12) (h3 : c % 15 = 13) : (a + b + c) % 15 = 3 := 
by
  sorry

end remainder_of_sum_l260_260094


namespace x_plus_y_equals_22_l260_260719

theorem x_plus_y_equals_22 (x y : ℕ) (h1 : 2^x = 4^(y + 2)) (h2 : 27^y = 9^(x - 7)) : x + y = 22 := 
sorry

end x_plus_y_equals_22_l260_260719


namespace weightlifting_winner_l260_260808

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l260_260808


namespace inscribed_circle_equals_arc_length_l260_260604

open Real

theorem inscribed_circle_equals_arc_length 
  (R : ℝ) 
  (hR : 0 < R) 
  (θ : ℝ)
  (hθ : θ = (2 * π) / 3)
  (r : ℝ)
  (h_r : r = R / 2) 
  : 2 * π * r = 2 * π * R * (θ / (2 * π)) := by
  sorry

end inscribed_circle_equals_arc_length_l260_260604


namespace students_taking_either_not_both_l260_260091

theorem students_taking_either_not_both (students_both : ℕ) (students_physics : ℕ) (students_only_chemistry : ℕ) :
  students_both = 12 →
  students_physics = 22 →
  students_only_chemistry = 9 →
  students_physics - students_both + students_only_chemistry = 19 :=
by
  intros h_both h_physics h_chemistry
  rw [h_both, h_physics, h_chemistry]
  repeat { sorry }

end students_taking_either_not_both_l260_260091


namespace exists_odd_k_l260_260584

noncomputable def f (n : ℕ) : ℕ :=
sorry

theorem exists_odd_k : 
  (∀ m n : ℕ, f (m * n) = f m * f n) → 
  (∀ m n : ℕ, (m + n) ∣ (f m + f n)) → 
  ∃ k : ℕ, (k % 2 = 1) ∧ (∀ n : ℕ, f n = n ^ k) :=
sorry

end exists_odd_k_l260_260584


namespace indefinite_integral_solution_l260_260100

open Real

theorem indefinite_integral_solution (c : ℝ) : 
  ∫ x, (1 - cos x) / (x - sin x) ^ 2 = - 1 / (x - sin x) + c := 
sorry

end indefinite_integral_solution_l260_260100


namespace purchasing_plan_exists_l260_260523

-- Define the structure for our purchasing plan
structure PurchasingPlan where
  n3 : ℕ
  n6 : ℕ
  n9 : ℕ
  n12 : ℕ
  n15 : ℕ
  n19 : ℕ
  n21 : ℕ
  n30 : ℕ

-- Define the length function to sum up the total length of the purchasing plan
def length (p : PurchasingPlan) : ℕ :=
  3 * p.n3 + 6 * p.n6 + 9 * p.n9 + 12 * p.n12 + 15 * p.n15 + 19 * p.n19 + 21 * p.n21 + 30 * p.n30

-- Define the purchasing options
def options : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

-- Define the requirement
def requiredLength : ℕ := 50

-- State the theorem that there exists a purchasing plan that sums up to the required length
theorem purchasing_plan_exists : ∃ p : PurchasingPlan, length p = requiredLength :=
  sorry

end purchasing_plan_exists_l260_260523


namespace sum_of_smallest_and_largest_even_l260_260306

theorem sum_of_smallest_and_largest_even (n : ℤ) (h : n + (n + 2) + (n + 4) = 1194) : n + (n + 4) = 796 :=
by
  sorry

end sum_of_smallest_and_largest_even_l260_260306


namespace determine_y_increase_volume_l260_260957

noncomputable def volume_increase_y (r h y : ℝ) : Prop :=
  (1/3) * Real.pi * (r + y)^2 * h = (1/3) * Real.pi * r^2 * (h + y)

theorem determine_y_increase_volume (y : ℝ) :
  volume_increase_y 5 12 y ↔ y = 31 / 12 :=
by
  sorry

end determine_y_increase_volume_l260_260957


namespace estimate_correctness_l260_260917

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end estimate_correctness_l260_260917


namespace copy_is_better_l260_260456

variable (α : ℝ)

noncomputable def p_random : ℝ := 1 / 2
noncomputable def I_mistake : ℝ := α
noncomputable def p_caught : ℝ := 1 / 10
noncomputable def I_caught : ℝ := 3 * α
noncomputable def p_neighbor_wrong : ℝ := 1 / 5
noncomputable def p_not_caught : ℝ := 9 / 10

theorem copy_is_better (α : ℝ) : 
  (12 * α / 25) < (α / 2) := by
  -- Proof goes here
  sorry

end copy_is_better_l260_260456


namespace exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l260_260173

theorem exist_colored_points_r_gt_pi_div_sqrt3 (r : ℝ) (hr : r > π / Real.sqrt 3) 
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

theorem exist_colored_points_r_gt_pi_div_2 (r : ℝ) (hr : r > π / 2)
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

end exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l260_260173


namespace Sarah_ate_one_apple_l260_260196

theorem Sarah_ate_one_apple:
  ∀ (total_apples apples_given_to_teachers apples_given_to_friends apples_left: ℕ), 
  total_apples = 25 →
  apples_given_to_teachers = 16 →
  apples_given_to_friends = 5 →
  apples_left = 3 →
  total_apples - (apples_given_to_teachers + apples_given_to_friends + apples_left) = 1 :=
by
  intros total_apples apples_given_to_teachers apples_given_to_friends apples_left
  intro ht ht gt hf
  sorry

end Sarah_ate_one_apple_l260_260196


namespace dwarf_heights_l260_260348

-- Define the heights of the dwarfs.
variables (F J M : ℕ)

-- Given conditions
def condition1 : Prop := J + F = M
def condition2 : Prop := M + F = J + 34
def condition3 : Prop := M + J = F + 72

-- Proof statement
theorem dwarf_heights
  (h1 : condition1 F J M)
  (h2 : condition2 F J M)
  (h3 : condition3 F J M) :
  F = 17 ∧ J = 36 ∧ M = 53 :=
by
  sorry

end dwarf_heights_l260_260348


namespace jar_water_transfer_l260_260384

theorem jar_water_transfer
  (C_x : ℝ) (C_y : ℝ)
  (h1 : C_y = 1/2 * C_x)
  (WaterInX : ℝ)
  (WaterInY : ℝ)
  (h2 : WaterInX = 1/2 * C_x)
  (h3 : WaterInY = 1/2 * C_y) :
  WaterInX + WaterInY = 3/4 * C_x :=
by
  sorry

end jar_water_transfer_l260_260384


namespace exists_coeff_less_than_neg_one_l260_260663

theorem exists_coeff_less_than_neg_one 
  (P : Polynomial ℤ)
  (h1 : P.eval 1 = 0)
  (h2 : P.eval 2 = 0) :
  ∃ i, P.coeff i < -1 := sorry

end exists_coeff_less_than_neg_one_l260_260663


namespace find_value_of_M_l260_260082

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l260_260082


namespace poly_diff_independent_of_x_l260_260286

theorem poly_diff_independent_of_x (x y: ℤ) (m n : ℤ) 
  (h1 : (1 - n = 0)) 
  (h2 : (m + 3 = 0)) :
  n - m = 4 := by
  sorry

end poly_diff_independent_of_x_l260_260286


namespace cost_per_tissue_l260_260531

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue_l260_260531


namespace find_y_value_l260_260880

theorem find_y_value : (15^3 * 7^4) / 5670 = 1428.75 := by
  sorry

end find_y_value_l260_260880


namespace no_integer_solutions_l260_260940

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 :=
by
  sorry

end no_integer_solutions_l260_260940


namespace gcd_three_numbers_4557_1953_5115_l260_260364

theorem gcd_three_numbers_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 := 
by 
  sorry

end gcd_three_numbers_4557_1953_5115_l260_260364


namespace sum_ratio_arithmetic_sequence_l260_260469

noncomputable def sum_of_arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem sum_ratio_arithmetic_sequence (S : ℕ → ℝ) (hS : ∀ n, S n = sum_of_arithmetic_sequence n)
  (h_cond : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
sorry

end sum_ratio_arithmetic_sequence_l260_260469


namespace true_proposition_l260_260553

-- Define the propositions p and q
def p : Prop := ∃ x0 : ℝ, x0 ^ 2 - x0 + 1 ≥ 0

def q : Prop := ∀ (a b : ℝ), a < b → 1 / a > 1 / b

-- Prove that p ∧ ¬q is true
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l260_260553


namespace x_div_11p_is_integer_l260_260915

theorem x_div_11p_is_integer (x p : ℕ) (h1 : x > 0) (h2 : Prime p) (h3 : x = 66) : ∃ k : ℤ, x / (11 * p) = k := by
  sorry

end x_div_11p_is_integer_l260_260915


namespace probability_two_red_marbles_l260_260388

theorem probability_two_red_marbles
  (red_marbles : ℕ)
  (white_marbles : ℕ)
  (total_marbles : ℕ)
  (prob_first_red : ℚ)
  (prob_second_red_after_first_red : ℚ)
  (combined_probability : ℚ) :
  red_marbles = 5 →
  white_marbles = 7 →
  total_marbles = 12 →
  prob_first_red = 5 / 12 →
  prob_second_red_after_first_red = 4 / 11 →
  combined_probability = 5 / 33 →
  combined_probability = prob_first_red * prob_second_red_after_first_red := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_two_red_marbles_l260_260388


namespace triangle_side_b_range_l260_260164

noncomputable def sin60 := Real.sin (Real.pi / 3)

theorem triangle_side_b_range (a b : ℝ) (A : ℝ)
  (ha : a = 2)
  (hA : A = 60 * Real.pi / 180)
  (h_2solutions : b * sin60 < a ∧ a < b) :
  (2 < b ∧ b < 4 * Real.sqrt 3 / 3) :=
by
  sorry

end triangle_side_b_range_l260_260164


namespace n_squared_plus_one_divides_n_plus_one_l260_260544

theorem n_squared_plus_one_divides_n_plus_one (n : ℕ) (h : n^2 + 1 ∣ n + 1) : n = 1 :=
by
  sorry

end n_squared_plus_one_divides_n_plus_one_l260_260544


namespace fill_in_square_l260_260004

theorem fill_in_square (x y : ℝ) (h : 4 * x^2 * (81 / 4 * x * y) = 81 * x^3 * y) : (81 / 4 * x * y) = (81 / 4 * x * y) :=
by
  sorry

end fill_in_square_l260_260004


namespace simplify_expression_1_combine_terms_l260_260671

variable (a b : ℝ)

-- Problem 1: Simplification
theorem simplify_expression_1 : 2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by 
  sorry

-- Problem 2: Combine like terms
theorem combine_terms : 3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 := by 
  sorry

end simplify_expression_1_combine_terms_l260_260671


namespace smallest_integer_solution_l260_260973

theorem smallest_integer_solution :
  ∃ y : ℤ, (5 / 8 < (y - 3) / 19) ∧ ∀ z : ℤ, (5 / 8 < (z - 3) / 19) → y ≤ z :=
sorry

end smallest_integer_solution_l260_260973


namespace right_triangle_sides_l260_260739

-- Definitions based on the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2
def perimeter (a b c : ℕ) : ℕ := a + b + c
def inscribed_circle_radius (a b c : ℕ) : ℕ := (a + b - c) / 2

-- The theorem statement
theorem right_triangle_sides (a b c : ℕ) 
  (h_perimeter : perimeter a b c = 40)
  (h_radius : inscribed_circle_radius a b c = 3)
  (h_right : is_right_triangle a b c) :
  (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
by sorry

end right_triangle_sides_l260_260739


namespace ann_frosting_time_l260_260933

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l260_260933


namespace number_of_terms_added_l260_260815

theorem number_of_terms_added (k : ℕ) (h : 1 ≤ k) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k :=
by sorry

end number_of_terms_added_l260_260815


namespace correct_propositions_are_123_l260_260883

theorem correct_propositions_are_123
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x-1) = -f x → f x = f (x-2))
  (h2 : ∀ x, f (1 - x) = f (x - 1) → f (1 - x) = -f x)
  (h3 : ∀ x, f (x) = -f (-x)) :
  (∀ x, f (x-1) = -f x → ∃ c, c * (f (1-1)) = -f x) ∧
  (∀ x, f (1 - x) = f (x - 1) → ∀ x, f x = f (-x)) ∧
  (∀ x, f (x-1) = -f x → ∀ x, f (x - 2) = f x) :=
sorry

end correct_propositions_are_123_l260_260883


namespace inequality_solution_l260_260615

theorem inequality_solution (x : ℝ) (h : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1)) : x > 1 := 
sorry

end inequality_solution_l260_260615


namespace min_value_of_x_l260_260884

theorem min_value_of_x (x : ℝ) (h : 2 * (x + 1) ≥ x + 1) : x ≥ -1 := sorry

end min_value_of_x_l260_260884


namespace evaluate_expression_l260_260177

def f (x : ℕ) : ℕ := 4 * x + 2
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_expression : f (g (f 3)) = 186 := 
by 
  sorry

end evaluate_expression_l260_260177


namespace can_cut_into_equal_parts_l260_260031

-- We assume the existence of a shape S and some grid G along with a function cut
-- that cuts the shape S along grid G lines and returns two parts.
noncomputable def Shape := Type
noncomputable def Grid := Type
noncomputable def cut (S : Shape) (G : Grid) : Shape × Shape := sorry

-- We assume a function superimpose that checks whether two shapes can be superimposed
noncomputable def superimpose (S1 S2 : Shape) : Prop := sorry

-- Assume the given shape S and grid G
variable (S : Shape) (G : Grid)

-- The question rewritten as a Lean statement
theorem can_cut_into_equal_parts : ∃ (S₁ S₂ : Shape), cut S G = (S₁, S₂) ∧ superimpose S₁ S₂ := sorry

end can_cut_into_equal_parts_l260_260031


namespace binom_n_n_minus_1_l260_260969

theorem binom_n_n_minus_1 (n : ℕ) (h : 0 < n) : (Nat.choose n (n-1)) = n :=
  sorry

end binom_n_n_minus_1_l260_260969


namespace find_r_l260_260178

theorem find_r (f g : ℝ → ℝ) (monic_f : ∀x, f x = (x - r - 2) * (x - r - 8) * (x - a))
  (monic_g : ∀x, g x = (x - r - 4) * (x - r - 10) * (x - b)) (h : ∀ x, f x - g x = r):
  r = 32 :=
by
  sorry

end find_r_l260_260178


namespace every_integer_appears_exactly_once_l260_260770

-- Define the sequence of integers
variable (a : ℕ → ℤ)

-- Define the conditions
axiom infinite_positives : ∀ n : ℕ, ∃ i > n, a i > 0
axiom infinite_negatives : ∀ n : ℕ, ∃ i > n, a i < 0
axiom distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n) ≠ (a j % n)

-- The proof statement
theorem every_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! i : ℕ, a i = x :=
sorry

end every_integer_appears_exactly_once_l260_260770


namespace problem_statement_l260_260136
noncomputable def a : ℕ := 10
noncomputable def b : ℕ := a^3

theorem problem_statement (a b : ℕ) (a_pos : 0 < a) (b_eq : b = a^3)
    (log_ab : Real.logb a (b : ℝ) = 3) (b_minus_a : b = a + 891) :
    a + b = 1010 :=
by
  sorry

end problem_statement_l260_260136


namespace remaining_credit_to_be_paid_l260_260588

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l260_260588


namespace Tigers_Sharks_min_games_l260_260605

open Nat

theorem Tigers_Sharks_min_games (N : ℕ) : 
  (let total_games := 3 + N
   let sharks_wins := 1 + N
   sharks_wins * 20 ≥ total_games * 19) ↔ N ≥ 37 := 
by
  sorry

end Tigers_Sharks_min_games_l260_260605


namespace tan_70_sin_80_eq_neg1_l260_260789

theorem tan_70_sin_80_eq_neg1 :
  (Real.tan 70 * Real.sin 80 * (Real.sqrt 3 * Real.tan 20 - 1) = -1) :=
sorry

end tan_70_sin_80_eq_neg1_l260_260789


namespace ellipse_slope_product_constant_l260_260279

noncomputable def ellipse_constant_slope_product (a b : ℝ) (P M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (N.1 = -M.1 ∧ N.2 = -M.2) ∧
  (∃ k_PM k_PN : ℝ, k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_PN = (P.2 - N.2) / (P.1 - N.1)) ∧
  ((P.2 - M.2) / (P.1 - M.1) * (P.2 - N.2) / (P.1 - N.1) = -b^2 / a^2)

theorem ellipse_slope_product_constant (a b : ℝ) (P M N : ℝ × ℝ) :
  ellipse_constant_slope_product a b P M N := 
sorry

end ellipse_slope_product_constant_l260_260279


namespace unique_identity_function_l260_260125

theorem unique_identity_function (f : ℕ+ → ℕ+) :
  (∀ (x y : ℕ+), 
    let a := x 
    let b := f y 
    let c := f (y + f x - 1)
    a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x, f x = x) :=
by
  intro h
  sorry

end unique_identity_function_l260_260125


namespace find_sum_l260_260488

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) + f (2 - x) = 0

theorem find_sum (f : ℝ → ℝ) (h_odd : odd_function f) (h_func : functional_equation f) (h_val : f 1 = 9) :
  f 2016 + f 2017 + f 2018 = 9 :=
  sorry

end find_sum_l260_260488


namespace coefficient_of_neg2ab_is_neg2_l260_260072

-- Define the term -2ab
def term : ℤ := -2

-- Define the function to get the coefficient from term -2ab
def coefficient (t : ℤ) : ℤ := t

-- The theorem stating the coefficient of -2ab is -2
theorem coefficient_of_neg2ab_is_neg2 : coefficient term = -2 :=
by
  -- Proof can be filled later
  sorry

end coefficient_of_neg2ab_is_neg2_l260_260072


namespace two_digit_number_problem_l260_260144

theorem two_digit_number_problem (a b : ℕ) :
  let M := 10 * b + a
  let N := 10 * a + b
  2 * M - N = 19 * b - 8 * a := by
  sorry

end two_digit_number_problem_l260_260144


namespace exponent_of_term_on_right_side_l260_260756

theorem exponent_of_term_on_right_side
  (s m : ℕ) 
  (h1 : (2^16) * (25^s) = 5 * (10^m))
  (h2 : m = 16) : m = 16 := 
by
  sorry

end exponent_of_term_on_right_side_l260_260756


namespace gcd_m_n_l260_260223

def m : ℕ := 131^2 + 243^2 + 357^2
def n : ℕ := 130^2 + 242^2 + 358^2

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l260_260223


namespace intercepts_congruence_l260_260231

theorem intercepts_congruence (m : ℕ) (h : m = 29) (x0 y0 : ℕ) (hx : 0 ≤ x0 ∧ x0 < m) (hy : 0 ≤ y0 ∧ y0 < m) 
  (h1 : 5 * x0 % m = (2 * 0 + 3) % m)  (h2 : (5 * 0) % m = (2 * y0 + 3) % m) : 
  x0 + y0 = 31 := by
  sorry

end intercepts_congruence_l260_260231


namespace max_magnitude_z3_plus_3z_plus_2i_l260_260587

open Complex

theorem max_magnitude_z3_plus_3z_plus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ M, M = 3 * Real.sqrt 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z^3 + 3 * z + 2 * Complex.I) ≤ M :=
by
  sorry

end max_magnitude_z3_plus_3z_plus_2i_l260_260587


namespace limit_of_function_l260_260099

open Real

theorem limit_of_function :
  (tendsto (λ x : ℝ, (6^(2 * x) - 7^(-2 * x)) / (sin (3 * x) - 2 * x)) (𝓝 0) (𝓝 (2 * log 42))) :=
sorry

end limit_of_function_l260_260099


namespace tim_income_percentage_less_than_juan_l260_260931

variables (M T J : ℝ)

theorem tim_income_percentage_less_than_juan 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.80 * J) : 
  100 - 100 * (T / J) = 50 :=
by
  sorry

end tim_income_percentage_less_than_juan_l260_260931


namespace students_attending_swimming_class_l260_260496

theorem students_attending_swimming_class 
  (total_students : ℕ) 
  (chess_percentage : ℕ) 
  (swimming_percentage : ℕ) 
  (number_of_students : ℕ)
  (chess_students := chess_percentage * total_students / 100)
  (swimming_students := swimming_percentage * chess_students / 100) 
  (condition1 : total_students = 2000)
  (condition2 : chess_percentage = 10)
  (condition3 : swimming_percentage = 50)
  (condition4 : number_of_students = chess_students) :
  swimming_students = 100 := 
by 
  sorry

end students_attending_swimming_class_l260_260496


namespace original_time_between_maintenance_checks_l260_260986

theorem original_time_between_maintenance_checks (x : ℝ) 
  (h1 : 2 * x = 60) : x = 30 := sorry

end original_time_between_maintenance_checks_l260_260986


namespace solution_set_inequality_l260_260436

variable (f : ℝ → ℝ)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

def condition_x_f_prime (f f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x^2 * f' x > 2 * x * f (-x)

-- Main theorem to prove the solution set of inequality
theorem solution_set_inequality (f' : ℝ → ℝ) :
  is_odd_function f →
  derivative f f' →
  condition_x_f_prime f f' →
  ∀ x : ℝ, x^2 * f x < (3 * x - 1)^2 * f (1 - 3 * x) → x < (1 / 4) := 
  by
    intros h_odd h_deriv h_cond x h_ineq
    sorry

end solution_set_inequality_l260_260436


namespace pascal_30th_31st_numbers_l260_260750

-- Definitions based on conditions
def pascal_triangle_row_34 (k : ℕ) : ℕ := Nat.choose 34 k

-- Problem statement in Lean 4: proving the equations
theorem pascal_30th_31st_numbers :
  pascal_triangle_row_34 29 = 278256 ∧
  pascal_triangle_row_34 30 = 46376 :=
by
  sorry

end pascal_30th_31st_numbers_l260_260750


namespace train_length_approx_90_l260_260681

noncomputable def speed_in_m_per_s := (124 : ℝ) * (1000 / 3600)

noncomputable def time_in_s := (2.61269421026963 : ℝ)

noncomputable def length_of_train := speed_in_m_per_s * time_in_s

theorem train_length_approx_90 : abs (length_of_train - 90) < 1e-9 :=
  by
  sorry

end train_length_approx_90_l260_260681


namespace weightlifting_winner_l260_260809

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l260_260809


namespace divisibility_l260_260473

theorem divisibility (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  (n^5 + 1) ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) := 
sorry

end divisibility_l260_260473


namespace fractional_part_lawn_remainder_l260_260341

def mary_mowing_time := 3 -- Mary can mow the lawn in 3 hours
def tom_mowing_time := 6  -- Tom can mow the lawn in 6 hours
def mary_working_hours := 1 -- Mary works for 1 hour alone

theorem fractional_part_lawn_remainder : 
  (1 - mary_working_hours / mary_mowing_time) = 2 / 3 := 
by
  sorry

end fractional_part_lawn_remainder_l260_260341


namespace smallest_n_satisfies_conditions_l260_260652

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260652


namespace shaded_area_in_octagon_l260_260310

theorem shaded_area_in_octagon (s r : ℝ) (h_s : s = 4) (h_r : r = s / 2) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_semicircles := 8 * (π * r^2 / 2)
  area_octagon - area_semicircles = 32 * (1 + Real.sqrt 2) - 16 * π := by
  sorry

end shaded_area_in_octagon_l260_260310


namespace packs_of_gum_bought_l260_260689

noncomputable def initial_amount : ℝ := 10.00
noncomputable def gum_cost : ℝ := 1.00
noncomputable def choc_bars : ℝ := 5.00
noncomputable def choc_bar_cost : ℝ := 1.00
noncomputable def candy_canes : ℝ := 2.00
noncomputable def candy_cane_cost : ℝ := 0.50
noncomputable def leftover_amount : ℝ := 1.00

theorem packs_of_gum_bought : (initial_amount - leftover_amount - (choc_bars * choc_bar_cost + candy_canes * candy_cane_cost)) / gum_cost = 3 :=
by
  sorry

end packs_of_gum_bought_l260_260689


namespace mary_money_left_l260_260048

variable (p : ℝ)

theorem mary_money_left :
  have cost_drinks := 3 * p
  have cost_medium_pizza := 2 * p
  have cost_large_pizza := 3 * p
  let total_cost := cost_drinks + cost_medium_pizza + cost_large_pizza
  30 - total_cost = 30 - 8 * p := by {
    sorry
  }

end mary_money_left_l260_260048


namespace comparison_among_abc_l260_260435

noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (1/5)^2
noncomputable def c : ℝ := Real.log (1/5) / Real.log 2

theorem comparison_among_abc : a > b ∧ b > c :=
by
  -- Assume the necessary conditions and the conclusion.
  sorry

end comparison_among_abc_l260_260435


namespace makenna_garden_larger_by_132_l260_260582

-- Define the dimensions of Karl's garden
def length_karl : ℕ := 22
def width_karl : ℕ := 50

-- Define the dimensions of Makenna's garden including the walking path
def length_makenna_total : ℕ := 30
def width_makenna_total : ℕ := 46
def walking_path_width : ℕ := 1

-- Define the area calculation functions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

-- Calculate the areas
def area_karl : ℕ := area length_karl width_karl
def area_makenna : ℕ := area (length_makenna_total - 2 * walking_path_width) (width_makenna_total - 2 * walking_path_width)

-- Define the theorem to prove
theorem makenna_garden_larger_by_132 :
  area_makenna = area_karl + 132 :=
by
  -- We skip the proof part
  sorry

end makenna_garden_larger_by_132_l260_260582


namespace total_crayons_l260_260855

-- Definitions for the conditions
def crayons_per_child : Nat := 12
def number_of_children : Nat := 18

-- The statement to be proved
theorem total_crayons :
  (crayons_per_child * number_of_children = 216) := 
by
  sorry

end total_crayons_l260_260855


namespace isosceles_right_triangle_leg_length_l260_260609

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_leg_length_l260_260609


namespace inequality_on_positive_reals_l260_260769

variable {a b c : ℝ}

theorem inequality_on_positive_reals (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_on_positive_reals_l260_260769


namespace probability_of_drawing_two_white_balls_l260_260575

-- Define the total number of balls and their colors
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

-- Define the total number of ways to draw 2 balls from 4
def total_draw_ways : ℕ := (total_balls.choose 2)

-- Define the number of ways to draw 2 white balls
def white_draw_ways : ℕ := (white_balls.choose 2)

-- Define the probability of drawing 2 white balls
def probability_white_draw : ℚ := white_draw_ways / total_draw_ways

-- The main theorem statement to prove
theorem probability_of_drawing_two_white_balls :
  probability_white_draw = 1 / 6 := by
  sorry

end probability_of_drawing_two_white_balls_l260_260575


namespace solution_set_ineq_l260_260710

theorem solution_set_ineq (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ 5 < x ∧ x ≤ 13 / 2 :=
sorry

end solution_set_ineq_l260_260710


namespace distance_car_to_stream_l260_260169

theorem distance_car_to_stream (total_distance : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ) (h1 : total_distance = 0.7) (h2 : stream_to_meadow = 0.4) (h3 : meadow_to_campsite = 0.1) :
  (total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2) :=
by
  sorry

end distance_car_to_stream_l260_260169


namespace parabola_sum_l260_260418

variables (a b c x y : ℝ)

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_sum (h1 : ∀ x, quadratic a b c x = -(x - 3)^2 + 4)
    (h2 : quadratic a b c 1 = 0)
    (h3 : quadratic a b c 5 = 0) :
    a + b + c = 0 :=
by
  -- We assume quadratic(a, b, c, x) = a * x^2 + b * x + c
  -- We assume quadratic(a, b, c, 1) = 0 and quadratic(a, b, c, 5) = 0
  -- We need to prove a + b + c = 0
  sorry

end parabola_sum_l260_260418


namespace planted_fraction_l260_260132

theorem planted_fraction (a b c : ℕ) (x h : ℝ) 
  (h_right_triangle : a = 5 ∧ b = 12)
  (h_hypotenuse : c = 13)
  (h_square_dist : x = 3) : 
  (h * ((a * b) - (x^2))) / (a * b / 2) = (7 : ℝ) / 10 :=
by
  sorry

end planted_fraction_l260_260132


namespace three_digit_square_ends_with_self_l260_260872

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l260_260872


namespace positive_difference_of_y_l260_260484

theorem positive_difference_of_y (y : ℝ) (h : (50 + y) / 2 = 35) : |50 - y| = 30 :=
by
  sorry

end positive_difference_of_y_l260_260484


namespace triangle_XYZ_median_inequalities_l260_260580

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end triangle_XYZ_median_inequalities_l260_260580


namespace MoneyDivision_l260_260246

theorem MoneyDivision (w x y z : ℝ)
  (hw : y = 0.5 * w)
  (hx : x = 0.7 * w)
  (hz : z = 0.3 * w)
  (hy : y = 90) :
  w + x + y + z = 450 := by
  sorry

end MoneyDivision_l260_260246


namespace kite_area_l260_260776

theorem kite_area (EF GH : ℝ) (FG EH : ℕ) (h1 : FG * FG + EH * EH = 25) : EF * GH = 12 :=
by
  sorry

end kite_area_l260_260776


namespace sum_not_fourteen_l260_260220

theorem sum_not_fourteen (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
  (hprod : a * b * c * d = 120) : a + b + c + d ≠ 14 :=
sorry

end sum_not_fourteen_l260_260220


namespace inequality_proof_l260_260153

variables {a b c : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c :=
by
  sorry

end inequality_proof_l260_260153


namespace sin_inequality_in_triangle_l260_260163

theorem sin_inequality_in_triangle (A B C : ℝ) (hA_leq_B : A ≤ B) (hB_leq_C : B ≤ C)
  (hSum : A + B + C = π) (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π) :
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := 
sorry

end sin_inequality_in_triangle_l260_260163


namespace chocolate_bars_cost_l260_260464

variable (n : ℕ) (c : ℕ)

-- Jessica's purchase details
def gummy_bears_packs := 10
def gummy_bears_cost_per_pack := 2
def chocolate_chips_bags := 20
def chocolate_chips_cost_per_bag := 5

-- Calculated costs
def total_gummy_bears_cost := gummy_bears_packs * gummy_bears_cost_per_pack
def total_chocolate_chips_cost := chocolate_chips_bags * chocolate_chips_cost_per_bag

-- Total cost
def total_cost := 150

-- Remaining cost for chocolate bars
def remaining_cost_for_chocolate_bars := total_cost - (total_gummy_bears_cost + total_chocolate_chips_cost)

theorem chocolate_bars_cost (h : remaining_cost_for_chocolate_bars = n * c) : remaining_cost_for_chocolate_bars = 30 :=
by
  sorry

end chocolate_bars_cost_l260_260464


namespace quadratic_value_at_point_l260_260108

variable (a b c : ℝ)

-- Given: A quadratic function f(x) = ax^2 + bx + c that passes through the point (3,10)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_value_at_point
  (h : f a b c 3 = 10) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end quadratic_value_at_point_l260_260108


namespace friend_charge_per_animal_l260_260592

-- Define the conditions.
def num_cats := 2
def num_dogs := 3
def total_payment := 65

-- Define the total number of animals.
def total_animals := num_cats + num_dogs

-- Define the charge per animal per night.
def charge_per_animal := total_payment / total_animals

-- State the theorem.
theorem friend_charge_per_animal : charge_per_animal = 13 := by
  -- Proof goes here.
  sorry

end friend_charge_per_animal_l260_260592


namespace evaluate_expression_l260_260856

theorem evaluate_expression : (502 * 502) - (501 * 503) = 1 := sorry

end evaluate_expression_l260_260856


namespace monotonicity_of_f_range_of_k_for_three_zeros_l260_260288

noncomputable def f (x k : ℝ) : ℝ := x^3 - k * x + k^2

def f_derivative (x k : ℝ) : ℝ := 3 * x^2 - k

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x : ℝ, 0 <= f_derivative x k) ↔ k <= 0 :=
by sorry

theorem range_of_k_for_three_zeros : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0) ↔ (0 < k ∧ k < 4 / 27) :=
by sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l260_260288


namespace car_catch_truck_l260_260836

theorem car_catch_truck (truck_speed car_speed : ℕ) (time_head_start : ℕ) (t : ℕ)
  (h1 : truck_speed = 45) (h2 : car_speed = 60) (h3 : time_head_start = 1) :
  45 * t + 45 = 60 * t → t = 3 := by
  intro h
  sorry

end car_catch_truck_l260_260836


namespace pencils_per_row_cannot_be_determined_l260_260708

theorem pencils_per_row_cannot_be_determined
  (rows : ℕ)
  (total_crayons : ℕ)
  (crayons_per_row : ℕ)
  (h_total_crayons: total_crayons = 210)
  (h_rows: rows = 7)
  (h_crayons_per_row: crayons_per_row = 30) :
  ∀ (pencils_per_row : ℕ), false :=
by
  sorry

end pencils_per_row_cannot_be_determined_l260_260708


namespace find_special_three_digit_numbers_l260_260870

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l260_260870


namespace set_union_example_l260_260301

theorem set_union_example (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) : M ∪ N = {1, 2, 3} := 
by
  sorry

end set_union_example_l260_260301


namespace find_sum_l260_260018

variable (a b : ℚ)

theorem find_sum :
  2 * a + 5 * b = 31 ∧ 4 * a + 3 * b = 35 → a + b = 68 / 7 := by
  sorry

end find_sum_l260_260018


namespace segments_to_start_l260_260198

-- Define the problem statement conditions in Lean 4
def concentric_circles : Prop := sorry -- Placeholder, as geometry involving tangents and arcs isn't directly supported

def chord_tangent_small_circle (AB : Prop) : Prop := sorry -- Placeholder, detailing tangency

def angle_ABC_eq_60 (A B C : Prop) : Prop := sorry -- Placeholder, situating angles in terms of Lean formalism

-- Proof statement
theorem segments_to_start (A B C : Prop) :
  concentric_circles →
  chord_tangent_small_circle (A ↔ B) →
  chord_tangent_small_circle (B ↔ C) →
  angle_ABC_eq_60 A B C →
  ∃ n : ℕ, n = 3 :=
sorry

end segments_to_start_l260_260198


namespace set_representation_l260_260158

def is_Natural (n : ℕ) : Prop :=
  n ≠ 0

def condition (x : ℕ) : Prop :=
  x^2 - 3*x < 0

theorem set_representation :
  {x : ℕ | condition x ∧ is_Natural x} = {1, 2} := 
sorry

end set_representation_l260_260158


namespace initial_amount_l260_260244

theorem initial_amount (M : ℝ) 
  (H1 : M * (2/3) * (4/5) * (3/4) * (5/7) * (5/6) = 200) : 
  M = 840 :=
by
  -- Proof to be provided
  sorry

end initial_amount_l260_260244


namespace total_flour_l260_260192

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l260_260192


namespace solve_for_t_l260_260162

theorem solve_for_t (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : x = y → t = 1/2 :=
by
  sorry

end solve_for_t_l260_260162


namespace pos_difference_between_highest_and_second_smallest_enrollment_l260_260505

def varsity_enrollment : ℕ := 1520
def northwest_enrollment : ℕ := 1430
def central_enrollment : ℕ := 1900
def greenbriar_enrollment : ℕ := 1850

theorem pos_difference_between_highest_and_second_smallest_enrollment :
  (central_enrollment - varsity_enrollment) = 380 := 
by 
  sorry

end pos_difference_between_highest_and_second_smallest_enrollment_l260_260505


namespace anna_age_l260_260138

-- Define the conditions as given in the problem
variable (x : ℕ)
variable (m n : ℕ)

-- Translate the problem statement into Lean
axiom perfect_square_condition : x - 4 = m^2
axiom perfect_cube_condition : x + 3 = n^3

-- The proof problem statement in Lean 4
theorem anna_age : x = 5 :=
by
  sorry

end anna_age_l260_260138


namespace statement_A_statement_B_statement_C_l260_260557

variables {p : ℝ} (hp : p > 0) (x0 y0 x1 y1 x2 y2 : ℝ)
variables (h_parabola : ∀ x y, y^2 = 2*p*x) 
variables (h_point_P : ∀ k m, y0 ≠ 0 ∧ x0 = k*y0 + m)

-- Statement A
theorem statement_A (hy0 : y0 = 0) : y1 * y2 = -2 * p * x0 :=
sorry

-- Statement B
theorem statement_B (hx0 : x0 = 0) : 1 / y1 + 1 / y2 = 1 / y0 :=
sorry

-- Statement C
theorem statement_C : (y0 - y1) * (y0 - y2) = y0^2 - 2 * p * x0 :=
sorry

end statement_A_statement_B_statement_C_l260_260557


namespace min_2x3y2z_l260_260180

noncomputable def min_value (x y z : ℝ) : ℝ := 2 * (x^3) * (y^2) * z

theorem min_2x3y2z (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h : (1/x) + (1/y) + (1/z) = 9) :
  min_value x y z = 2 / 675 :=
sorry

end min_2x3y2z_l260_260180


namespace leg_length_of_isosceles_right_triangle_l260_260611

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * real.sqrt 2

def median_to_hypotenuse (a b c m : ℝ) : Prop :=
  is_isosceles_right_triangle a b c ∧ m = c / 2

-- The proof problem statement
theorem leg_length_of_isosceles_right_triangle (a b c m : ℝ) (h1 : is_isosceles_right_triangle a b c)
  (h2 : median_to_hypotenuse a b c m) (h3 : m = 12) : a = 12 * real.sqrt 2 :=
by
  sorry

end leg_length_of_isosceles_right_triangle_l260_260611


namespace find_teaspoons_of_salt_l260_260034

def sodium_in_salt (S : ℕ) : ℕ := 50 * S
def sodium_in_parmesan (P : ℕ) : ℕ := 25 * P

-- Initial total sodium amount with 8 ounces of parmesan
def initial_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 8

-- Reduced sodium after removing 4 ounces of parmesan
def reduced_sodium (S : ℕ) : ℕ := initial_total_sodium S * 2 / 3

-- Reduced sodium with 4 fewer ounces of parmesan cheese
def new_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 4

theorem find_teaspoons_of_salt : ∃ (S : ℕ), reduced_sodium S = new_total_sodium S ∧ S = 2 :=
by
  sorry

end find_teaspoons_of_salt_l260_260034


namespace intersection_eq_l260_260735

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by sorry

end intersection_eq_l260_260735


namespace total_kids_at_camp_l260_260703

-- Definition of the conditions
def kids_from_lawrence_camp : ℕ := 34044
def kids_from_outside_camp : ℕ := 424944

-- The proof statement
theorem total_kids_at_camp : kids_from_lawrence_camp + kids_from_outside_camp = 459988 := by
  sorry

end total_kids_at_camp_l260_260703


namespace equal_work_women_l260_260982

-- Let W be the amount of work one woman can do in a day.
-- Let M be the amount of work one man can do in a day.
-- Let x be the number of women who do the same amount of work as 5 men.

def numWomenEqualWork (W : ℝ) (M : ℝ) (x : ℝ) : Prop :=
  5 * M = x * W

theorem equal_work_women (W M x : ℝ) 
  (h1 : numWomenEqualWork W M x)
  (h2 : (3 * M + 5 * W) * 10 = (7 * W) * 14) :
  x = 8 :=
sorry

end equal_work_women_l260_260982


namespace team_a_builds_per_day_l260_260373

theorem team_a_builds_per_day (x : ℝ) (h1 : (150 / x = 100 / (2 * x - 30))) : x = 22.5 := by
  sorry

end team_a_builds_per_day_l260_260373


namespace triangle_side_b_l260_260574

open Real

variable {a b c : ℝ} (A B C : ℝ)

theorem triangle_side_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin B = 6 * cos A * sin C) : b = 3 :=
sorry

end triangle_side_b_l260_260574


namespace probability_at_least_seven_heads_or_tails_l260_260261

open Nat

-- Define the probability of getting at least seven heads or tails in eight coin flips
theorem probability_at_least_seven_heads_or_tails :
  let total_outcomes := 2^8
  let favorable_outcomes := (choose 8 7) + (choose 8 7) + 1 + 1
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 9 / 128 := by
  sorry

end probability_at_least_seven_heads_or_tails_l260_260261


namespace surface_area_hemisphere_l260_260977

theorem surface_area_hemisphere
  (r : ℝ)
  (h₁ : 4 * Real.pi * r^2 = 4 * Real.pi * r^2)
  (h₂ : Real.pi * r^2 = 3) :
  3 * Real.pi * r^2 = 9 :=
by
  sorry

end surface_area_hemisphere_l260_260977


namespace baseball_card_difference_l260_260339

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) (h1 : marcus_cards = 210) (h2 : carter_cards = 152) : marcus_cards - carter_cards = 58 :=
by {
    --skip the proof
    sorry
}

end baseball_card_difference_l260_260339


namespace wax_current_eq_l260_260000

-- Define the constants for the wax required and additional wax needed
def w_required : ℕ := 166
def w_more : ℕ := 146

-- Define the term to represent the current wax he has
def w_current : ℕ := w_required - w_more

-- Theorem statement to prove the current wax quantity
theorem wax_current_eq : w_current = 20 := by
  -- Proof outline would go here, but per instructions, we skip with sorry
  sorry

end wax_current_eq_l260_260000


namespace geometric_sequence_sum_l260_260433

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 3)
  (h_sum : a 1 + a 3 + a 5 = 21) : 
  a 3 + a 5 + a 7 = 42 :=
sorry

end geometric_sequence_sum_l260_260433


namespace unique_real_solution_l260_260860

noncomputable def cubic_eq (b x : ℝ) : ℝ :=
  x^3 - b * x^2 - 3 * b * x + b^2 - 2

theorem unique_real_solution (b : ℝ) :
  (∃! x : ℝ, cubic_eq b x = 0) ↔ b = 7 / 4 :=
by
  sorry

end unique_real_solution_l260_260860


namespace smallest_rectangles_cover_square_l260_260380

theorem smallest_rectangles_cover_square :
  ∃ (n : ℕ), n = 8 ∧ ∀ (a : ℕ), ∀ (b : ℕ), (a = 2) ∧ (b = 4) → 
  ∃ (s : ℕ), s = 8 ∧ (s * s) / (a * b) = n :=
by
  sorry

end smallest_rectangles_cover_square_l260_260380


namespace minimize_expression_l260_260226

theorem minimize_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  exists 3
  split
  -- Here, you would prove the inequality and the fact that x = 3 gives the minimum.
  sorry

end minimize_expression_l260_260226


namespace three_digit_numbers_with_square_ending_in_them_l260_260865

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l260_260865


namespace find_ABC_base10_l260_260068

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10_l260_260068


namespace contracting_arrangements_1680_l260_260532

def num_contracting_arrangements (n a b c d : ℕ) : ℕ :=
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c

theorem contracting_arrangements_1680 : num_contracting_arrangements 8 3 1 2 2 = 1680 := by
  unfold num_contracting_arrangements
  simp
  sorry

end contracting_arrangements_1680_l260_260532


namespace least_positive_integer_div_conditions_l260_260971

theorem least_positive_integer_div_conditions :
  ∃ n > 1, (n % 4 = 3) ∧ (n % 5 = 3) ∧ (n % 7 = 3) ∧ (n % 10 = 3) ∧ (n % 11 = 3) ∧ n = 1543 := 
by 
  sorry

end least_positive_integer_div_conditions_l260_260971


namespace N2O3_weight_l260_260819

-- Definitions from the conditions
def molecularWeightN : Float := 14.01
def molecularWeightO : Float := 16.00
def molecularWeightN2O3 : Float := (2 * molecularWeightN) + (3 * molecularWeightO)
def moles : Float := 4

-- The main proof problem statement
theorem N2O3_weight (h1 : molecularWeightN = 14.01)
                    (h2 : molecularWeightO = 16.00)
                    (h3 : molecularWeightN2O3 = (2 * molecularWeightN) + (3 * molecularWeightO))
                    (h4 : moles = 4) :
                    (moles * molecularWeightN2O3) = 304.08 :=
by
  sorry

end N2O3_weight_l260_260819


namespace train_cable_car_distance_and_speeds_l260_260678
-- Import necessary libraries

-- Defining the variables and conditions
variables (s v1 v2 : ℝ)
variables (half_hour_sym_dist additional_distance quarter_hour_meet : ℝ)

-- Defining the conditions
def conditions :=
  (half_hour_sym_dist = v1 * (1 / 2) + v2 * (1 / 2)) ∧
  (additional_distance = 2 / v2) ∧
  (quarter_hour_meet = 1 / 4) ∧
  (v1 + v2 = 2 * s) ∧
  (v2 * (additional_distance + half_hour_sym_dist) = (v1 * (additional_distance + half_hour_sym_dist) - s)) ∧
  ((v1 + v2) * (half_hour_sym_dist + additional_distance + quarter_hour_meet) = 2 * s)

-- Proving the statement
theorem train_cable_car_distance_and_speeds
  (h : conditions s v1 v2 half_hour_sym_dist additional_distance quarter_hour_meet) :
  s = 24 ∧ v1 = 40 ∧ v2 = 8 := sorry

end train_cable_car_distance_and_speeds_l260_260678


namespace solve_inequality_l260_260356

theorem solve_inequality (x : ℝ) : 
  (-1 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧ 
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 1) ↔ (1 < x) := 
by 
  sorry

end solve_inequality_l260_260356


namespace area_of_sector_l260_260019

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l260_260019


namespace clara_total_cookies_l260_260403

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l260_260403


namespace largest_side_of_enclosure_l260_260343

-- Definitions for the conditions
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem largest_side_of_enclosure (l w : ℝ) (h_fencing : perimeter l w = 240) (h_area : area l w = 12 * 240) : l = 86.83 ∨ w = 86.83 :=
by {
  sorry
}

end largest_side_of_enclosure_l260_260343


namespace arithmetic_sequence_terms_l260_260305

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h2 : a 1 + a 2 + a 3 = 34)
  (h3 : a n + a (n-1) + a (n-2) = 146)
  (h4 : S n = 390)
  (h5 : ∀ i j, a i + a j = a (i+1) + a (j-1)) :
  n = 13 :=
sorry

end arithmetic_sequence_terms_l260_260305


namespace boat_speed_in_still_water_l260_260514

theorem boat_speed_in_still_water : 
  ∀ (V_b V_s : ℝ), 
  V_b + V_s = 15 → 
  V_b - V_s = 5 → 
  V_b = 10 :=
by
  intros V_b V_s h1 h2
  have h3 : 2 * V_b = 20 := by linarith
  linarith

end boat_speed_in_still_water_l260_260514


namespace uncle_age_when_seokjin_is_12_l260_260943

-- Definitions for the conditions
def mother_age_when_seokjin_born : ℕ := 32
def uncle_is_younger_by : ℕ := 3
def seokjin_age : ℕ := 12

-- Definition for the main hypothesis
theorem uncle_age_when_seokjin_is_12 :
  let mother_age_when_seokjin_is_12 := mother_age_when_seokjin_born + seokjin_age
  let uncle_age_when_seokjin_is_12 := mother_age_when_seokjin_is_12 - uncle_is_younger_by
  uncle_age_when_seokjin_is_12 = 41 :=
by
  sorry

end uncle_age_when_seokjin_is_12_l260_260943


namespace find_missing_digit_l260_260950

theorem find_missing_digit 
  (x : Nat) 
  (h : 16 + x ≡ 0 [MOD 9]) : 
  x = 2 :=
sorry

end find_missing_digit_l260_260950


namespace smallest_n_satisfies_conditions_l260_260649

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260649


namespace number_of_lattice_points_in_triangle_l260_260115

theorem number_of_lattice_points_in_triangle (L : ℕ) (hL : L > 1) :
  ∃ I, I = (L^2 - 1) / 2 :=
by
  sorry

end number_of_lattice_points_in_triangle_l260_260115


namespace elizabeth_fruits_l260_260538

def total_fruits (initial_bananas initial_apples initial_grapes eaten_bananas eaten_apples eaten_grapes : Nat) : Nat :=
  let bananas_left := initial_bananas - eaten_bananas
  let apples_left := initial_apples - eaten_apples
  let grapes_left := initial_grapes - eaten_grapes
  bananas_left + apples_left + grapes_left

theorem elizabeth_fruits : total_fruits 12 7 19 4 2 10 = 22 := by
  sorry

end elizabeth_fruits_l260_260538


namespace at_least_one_nonnegative_l260_260146

theorem at_least_one_nonnegative
  (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) (h4 : a4 ≠ 0)
  (h5 : a5 ≠ 0) (h6 : a6 ≠ 0) (h7 : a7 ≠ 0) (h8 : a8 ≠ 0)
  : (a1 * a3 + a2 * a4 ≥ 0) ∨ (a1 * a5 + a2 * a6 ≥ 0) ∨ (a1 * a7 + a2 * a8 ≥ 0) ∨
    (a3 * a5 + a4 * a6 ≥ 0) ∨ (a3 * a7 + a4 * a8 ≥ 0) ∨ (a5 * a7 + a6 * a8 ≥ 0) := 
sorry

end at_least_one_nonnegative_l260_260146


namespace price_second_oil_per_litre_is_correct_l260_260980

-- Definitions based on conditions
def price_first_oil_per_litre := 54
def volume_first_oil := 10
def volume_second_oil := 5
def mixture_rate_per_litre := 58
def total_volume := volume_first_oil + volume_second_oil
def total_cost_mixture := total_volume * mixture_rate_per_litre
def total_cost_first_oil := volume_first_oil * price_first_oil_per_litre

-- The statement to prove
theorem price_second_oil_per_litre_is_correct (x : ℕ) (h : total_cost_first_oil + (volume_second_oil * x) = total_cost_mixture) : x = 66 :=
by
  sorry

end price_second_oil_per_litre_is_correct_l260_260980


namespace roots_relationship_l260_260597

variable {a b c : ℝ} (h : a ≠ 0)

theorem roots_relationship (x y : ℝ) :
  (x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (y = (-b + Real.sqrt (b^2 - 4*a*c)) / 2 ∨ y = (-b - Real.sqrt (b^2 - 4*a*c)) / 2) →
  (x = y / a) :=
by
  sorry

end roots_relationship_l260_260597


namespace num_mappings_from_A_to_A_is_4_l260_260491

-- Define the number of elements in set A
def set_A_card := 2

-- Define the proof problem
theorem num_mappings_from_A_to_A_is_4 (h : set_A_card = 2) : (set_A_card ^ set_A_card) = 4 :=
by
  sorry

end num_mappings_from_A_to_A_is_4_l260_260491


namespace taller_tree_height_l260_260358

variable (T S : ℝ)

theorem taller_tree_height (h1 : T - S = 20)
  (h2 : T - 10 = 3 * (S - 10)) : T = 40 :=
sorry

end taller_tree_height_l260_260358


namespace arithmetic_geometric_sequence_l260_260898

theorem arithmetic_geometric_sequence
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 3 = 12)
  (h3 : (a 1 + (a 2 - a 1))^2 = a 1 * (a 1 + 2 * (a 2 - a 1) + 2))
  (h4 : ∀ n, b n = (3 ^ n) * a n) :
  (∀ n, a n = 2 * n) ∧ 
  (∀ n, T n = (2 * n - 1) * 3^(n + 1) / 2 + 3 / 2) :=
sorry

end arithmetic_geometric_sequence_l260_260898


namespace number_of_people_only_went_to_aquarium_is_5_l260_260114

-- Define the conditions
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the problem in Lean
theorem number_of_people_only_went_to_aquarium_is_5 :
  ∃ x : ℕ, (total_earnings - (group_size * (admission_fee + tour_fee)) = x * admission_fee) → x = 5 :=
by
  sorry

end number_of_people_only_went_to_aquarium_is_5_l260_260114


namespace tom_reads_700_pages_in_7_days_l260_260500

theorem tom_reads_700_pages_in_7_days
  (total_hours : ℕ)
  (total_days : ℕ)
  (pages_per_hour : ℕ)
  (reads_same_amount_every_day : Prop)
  (h1 : total_hours = 10)
  (h2 : total_days = 5)
  (h3 : pages_per_hour = 50)
  (h4 : reads_same_amount_every_day) :
  (total_hours / total_days) * (pages_per_hour * 7) = 700 :=
by
  -- Begin and skip proof with sorry
  sorry

end tom_reads_700_pages_in_7_days_l260_260500


namespace rectangle_area_l260_260515

theorem rectangle_area (r l b : ℝ) (h1: r = 30) (h2: l = (2 / 5) * r) (h3: b = 10) : 
  l * b = 120 := 
by
  sorry

end rectangle_area_l260_260515


namespace smallest_positive_n_l260_260645

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260645


namespace product_discount_l260_260245

theorem product_discount (P : ℝ) (h₁ : P > 0) :
  let price_after_first_discount := 0.7 * P
  let price_after_second_discount := 0.8 * price_after_first_discount
  let total_reduction := P - price_after_second_discount
  let percent_reduction := (total_reduction / P) * 100
  percent_reduction = 44 :=
by
  sorry

end product_discount_l260_260245


namespace A_not_losing_prob_correct_l260_260227

def probability_draw : ℚ := 1 / 2
def probability_A_wins : ℚ := 1 / 3
def probability_A_not_losing : ℚ := 5 / 6

theorem A_not_losing_prob_correct : 
  probability_draw + probability_A_wins = probability_A_not_losing := 
by sorry

end A_not_losing_prob_correct_l260_260227


namespace sales_volume_expression_reduction_for_desired_profit_l260_260676

-- Initial conditions definitions.
def initial_purchase_price : ℝ := 3
def initial_selling_price : ℝ := 5
def initial_sales_volume : ℝ := 100
def sales_increase_per_0_1_yuan : ℝ := 20
def desired_profit : ℝ := 300
def minimum_sales_volume : ℝ := 220

-- Question (1): Sales Volume Expression
theorem sales_volume_expression (x : ℝ) : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) = 100 + 200 * x :=
by sorry

-- Question (2): Determine Reduction for Desired Profit and Minimum Sales Volume
theorem reduction_for_desired_profit (x : ℝ) 
  (hx : (initial_selling_price - initial_purchase_price - x) * (initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x)) = desired_profit)
  (hy : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) >= minimum_sales_volume) :
  x = 1 :=
by sorry

end sales_volume_expression_reduction_for_desired_profit_l260_260676


namespace total_earnings_of_a_b_c_l260_260666

theorem total_earnings_of_a_b_c 
  (days_a days_b days_c : ℕ)
  (ratio_a ratio_b ratio_c : ℕ)
  (wage_c : ℕ) 
  (h_ratio : ratio_a * wage_c = 3 * (3 + 4 + 5))
  (h_ratio_a_b : ratio_b = 4 * wage_c / 5 * ratio_a / 60)
  (h_ratio_b_c : ratio_b = 4 * wage_c / 5 * ratio_c / 60):
  (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c) = 1480 := 
  by
    sorry

end total_earnings_of_a_b_c_l260_260666


namespace monotonicity_of_f_range_of_k_for_three_zeros_l260_260289

noncomputable def f (x k : ℝ) := x^3 - k * x + k^2

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_of_f (k : ℝ) :
  (k ≤ 0 → ∀ x y : ℝ, x ≤ y → f x k ≤ f y k) ∧ 
  (k > 0 → (∀ x : ℝ, x < -sqrt (k / 3) → f x k < f (-sqrt (k / 3)) k) ∧ 
            (∀ x : ℝ, x > sqrt (k / 3) → f x k > f (sqrt (k / 3)) k) ∧
            (f (-sqrt (k / 3)) k > f (sqrt (k / 3)) k)) :=
  sorry

-- Problem 2: Range of k for f(x) to have three zeros
theorem range_of_k_for_three_zeros (k : ℝ) : 
  (∃ a b c : ℝ, f a k = 0 ∧ f b k = 0 ∧ f c k = 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) ↔ (0 < k ∧ k < 4 / 27) :=
  sorry

end monotonicity_of_f_range_of_k_for_three_zeros_l260_260289


namespace profit_margin_comparison_l260_260948

theorem profit_margin_comparison
    (cost_price_A : ℝ) (selling_price_A : ℝ)
    (cost_price_B : ℝ) (selling_price_B : ℝ)
    (h1 : cost_price_A = 1600)
    (h2 : selling_price_A = 0.9 * 2000)
    (h3 : cost_price_B = 320)
    (h4 : selling_price_B = 0.8 * 460) :
    ((selling_price_B - cost_price_B) / cost_price_B) > ((selling_price_A - cost_price_A) / cost_price_A) := 
by
    sorry

end profit_margin_comparison_l260_260948


namespace wages_of_one_man_l260_260826

variable (R : Type) [DivisionRing R] [DecidableEq R]
variable (money : R)
variable (num_men : ℕ := 5)
variable (num_women : ℕ := 8)
variable (total_wages : R := 180)
variable (wages_men : R := 36)

axiom equal_women : num_men = num_women
axiom total_earnings (wages : ℕ → R) :
  (wages num_men) + (wages num_women) + (wages 8) = total_wages

theorem wages_of_one_man :
  wages_men = total_wages / num_men := by
  sorry

end wages_of_one_man_l260_260826


namespace total_tissues_brought_l260_260954

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l260_260954


namespace plates_used_l260_260342

theorem plates_used (P : ℕ) (h : 3 * 2 * P + 4 * 8 = 38) : P = 1 := by
  sorry

end plates_used_l260_260342


namespace total_cost_of_dresses_l260_260014

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l260_260014


namespace total_teeth_cleaned_l260_260507

/-
  Given:
   1. Dogs have 42 teeth.
   2. Cats have 30 teeth.
   3. Pigs have 28 teeth.
   4. There are 5 dogs.
   5. There are 10 cats.
   6. There are 7 pigs.
  Prove: The total number of teeth Vann will clean today is 706.
-/

theorem total_teeth_cleaned :
  let dogs: Nat := 5
  let cats: Nat := 10
  let pigs: Nat := 7
  let dog_teeth: Nat := 42
  let cat_teeth: Nat := 30
  let pig_teeth: Nat := 28
  (dogs * dog_teeth) + (cats * cat_teeth) + (pigs * pig_teeth) = 706 := by
  -- Proof goes here
  sorry

end total_teeth_cleaned_l260_260507


namespace trigonometric_identity_l260_260431

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 2 / 7 :=
by
  sorry

end trigonometric_identity_l260_260431


namespace arithmetic_progression_rth_term_l260_260716

theorem arithmetic_progression_rth_term (S : ℕ → ℕ) (hS : ∀ n, S n = 5 * n + 4 * n ^ 2) 
  (r : ℕ) : S r - S (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l260_260716


namespace total_students_l260_260213

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end total_students_l260_260213


namespace classroom_students_count_l260_260853

-- Definitions of given conditions
def total_students : ℕ := 1260

def aud_students : ℕ := (7 * total_students) / 18

def non_aud_students : ℕ := total_students - aud_students

def classroom_students : ℕ := (6 * non_aud_students) / 11

-- Theorem statement
theorem classroom_students_count : classroom_students = 420 := by
  sorry

end classroom_students_count_l260_260853


namespace yuna_solved_problems_l260_260096

def yuna_problems_per_day : ℕ := 8
def days_per_week : ℕ := 7
def yuna_weekly_problems : ℕ := 56

theorem yuna_solved_problems :
  yuna_problems_per_day * days_per_week = yuna_weekly_problems := by
  sorry

end yuna_solved_problems_l260_260096


namespace number_of_ways_to_distribute_balls_l260_260566

theorem number_of_ways_to_distribute_balls :
  (finset.card ((finset.range 8).powerset.filter (λ s, finset.card s ≤ 7)) / 2) = 64 :=
by sorry

end number_of_ways_to_distribute_balls_l260_260566


namespace audio_space_per_hour_l260_260238

/-
The digital music library holds 15 days of music.
The library occupies 20,000 megabytes of disk space.
The library contains both audio and video files.
Video files take up twice as much space per hour as audio files.
There is an equal number of hours for audio and video.
-/

theorem audio_space_per_hour (total_days : ℕ) (total_space : ℕ) (equal_hours : Prop) (video_space : ℕ → ℕ) 
  (H1 : total_days = 15)
  (H2 : total_space = 20000)
  (H3 : equal_hours)
  (H4 : ∀ x, video_space x = 2 * x) :
  ∃ x, x = 37 :=
by
  sorry

end audio_space_per_hour_l260_260238


namespace y_increase_for_x_increase_l260_260892

theorem y_increase_for_x_increase (x y : ℝ) (h : 4 * y = 9) : 12 * y = 27 :=
by
  sorry

end y_increase_for_x_increase_l260_260892


namespace sum_even_and_multiples_of_5_l260_260767

def num_even_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 5 -- even digits: {0, 2, 4, 6, 8}
  thousands * hundreds * tens * units

def num_multiples_of_5_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 2 -- multiples of 5 digits: {0, 5}
  thousands * hundreds * tens * units

theorem sum_even_and_multiples_of_5 : num_even_four_digit + num_multiples_of_5_four_digit = 6300 := by
  sorry

end sum_even_and_multiples_of_5_l260_260767


namespace find_smallest_n_l260_260639

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260639


namespace unique_prime_sum_and_diff_l260_260709

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def is_sum_of_two_primes (p : ℕ) : Prop :=
  ∃ q1 q2 : ℕ, is_prime q1 ∧ is_prime q2 ∧ p = q1 + q2

noncomputable def is_diff_of_two_primes (p : ℕ) : Prop :=
  ∃ q3 q4 : ℕ, is_prime q3 ∧ is_prime q4 ∧ q3 > q4 ∧ p = q3 - q4

theorem unique_prime_sum_and_diff :
  ∀ p : ℕ, is_prime p ∧ is_sum_of_two_primes p ∧ is_diff_of_two_primes p ↔ p = 5 := 
by
  sorry

end unique_prime_sum_and_diff_l260_260709


namespace circuit_boards_fail_inspection_l260_260165

theorem circuit_boards_fail_inspection (P F : ℝ) (h1 : P + F = 3200)
    (h2 : (1 / 8) * P + F = 456) : F = 64 :=
by
  sorry

end circuit_boards_fail_inspection_l260_260165


namespace common_value_of_4a_and_5b_l260_260908

theorem common_value_of_4a_and_5b (a b C : ℝ) (h1 : 4 * a = C) (h2 : 5 * b = C) (h3 : 40 * a * b = 1800) :
  C = 60 :=
sorry

end common_value_of_4a_and_5b_l260_260908


namespace knights_count_l260_260320

theorem knights_count (n : ℕ) (h : n = 65) : 
  ∃ k, k = 23 ∧ (∀ i, 1 ≤ i ∧ i ≤ n → (i.odd ↔ i ≥ 21)) :=
by
  exists 23
  sorry

end knights_count_l260_260320


namespace cube_positive_integers_solution_l260_260267

theorem cube_positive_integers_solution (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∃ k : ℕ, 2^(Nat.factorial a) + 2^(Nat.factorial b) + 2^(Nat.factorial c) = k^3) ↔ 
    ( (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
      (a = 1 ∧ b = 2 ∧ c = 1) ∨ 
      (a = 2 ∧ b = 1 ∧ c = 1) ) :=
by
  sorry

end cube_positive_integers_solution_l260_260267


namespace three_digit_square_ends_with_self_l260_260873

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l260_260873


namespace watermelon_heavier_than_pineapple_l260_260804

noncomputable def watermelon_weight : ℕ := 1 * 1000 + 300 -- Weight of one watermelon in grams
noncomputable def pineapple_weight : ℕ := 450 -- Weight of one pineapple in grams

theorem watermelon_heavier_than_pineapple :
    (4 * watermelon_weight = 5 * 1000 + 200) →
    (3 * watermelon_weight + 4 * pineapple_weight = 5 * 1000 + 700) →
    watermelon_weight - pineapple_weight = 850 :=
by
    intros h1 h2
    sorry

end watermelon_heavier_than_pineapple_l260_260804


namespace maximum_xyz_l260_260329

-- Given conditions
variables {x y z : ℝ}

-- Lean 4 statement with the conditions
theorem maximum_xyz (h₁ : x * y + 2 * z = (x + z) * (y + z))
  (h₂ : x + y + 2 * z = 2)
  (h₃ : 0 < x) (h₄ : 0 < y) (h₅ : 0 < z) :
  xyz = 0 :=
sorry

end maximum_xyz_l260_260329


namespace eval_sqrt_pow_l260_260412

theorem eval_sqrt_pow (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 12) :
  (real.sqrt ^ 4 (a ^ b)) ^ c = 4096 :=
by sorry

end eval_sqrt_pow_l260_260412


namespace solve_equation_l260_260965

def f (x : ℝ) := |3 * x - 2|

theorem solve_equation 
  (x : ℝ) 
  (a : ℝ)
  (hx1 : x ≠ 3)
  (hx2 : x ≠ 0) :
  (3 * x - 2) ^ 2 = (x + a) ^ 2 ↔
  (a = -4 * x + 2) ∨ (a = 2 * x - 2) := by
  sorry

end solve_equation_l260_260965


namespace ellipse_eq_form_l260_260728

noncomputable def ellipse_c (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ a = sqrt 2 ∧ b = 1 ∧ (e : ℝ) = sqrt 2 / 2 ∧ 
  P.x = 1 ∧ P.y = 1 / sqrt 2 ∧ |O - P| = sqrt 6 / 2

noncomputable def equation_of_line (k : ℝ) : Prop :=
  exists (a b : ℝ), a > b ∧ b > 0 ∧ 
  (a = sqrt 2 ∧ b = 1 ∧ (e : ℝ) = sqrt 2 / 2 ∧ 
  ( |O - P| = sqrt 6 / 2 ∧ 
    M.x = P.x / sqrt 2 / 2) ∧ 
    ((S : ℝ) = sqrt 2 / 2 ∧ (S_form: Formula.)triangle_area ((A : ℝ) + A.y k P = B) ∧ 2 / (1 + k^2) sqrt(1 + k^2) - intersect AOB = x

theorem ellipse_eq_form (C : ℝ): ellipse_c -> equation_of_line ->
 Prop = sorry

end ellipse_eq_form_l260_260728


namespace alcohol_percentage_new_mixture_l260_260386

theorem alcohol_percentage_new_mixture :
  let initial_alcohol_percentage := 0.90
  let initial_solution_volume := 24
  let added_water_volume := 16
  let total_new_volume := initial_solution_volume + added_water_volume
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_percentage
  let new_alcohol_percentage := (initial_alcohol_amount / total_new_volume) * 100
  new_alcohol_percentage = 54 := by
    sorry

end alcohol_percentage_new_mixture_l260_260386


namespace exponent_logarithm_simplifies_l260_260119

theorem exponent_logarithm_simplifies :
  (1/2 : ℝ) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 :=
by sorry

end exponent_logarithm_simplifies_l260_260119


namespace solution_l260_260287

def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^5 + p*x^3 + q*x - 8

theorem solution (p q : ℝ) (h : f (-2) p q = 10) : f 2 p q = -26 := by
  sorry

end solution_l260_260287


namespace ratio_of_product_of_composites_l260_260271

theorem ratio_of_product_of_composites :
  let A := [4, 6, 8, 9, 10, 12]
  let B := [14, 15, 16, 18, 20, 21]
  (A.foldl (λ x y => x * y) 1) / (B.foldl (λ x y => x * y) 1) = 1 / 49 :=
by
  -- Proof will be filled here
  sorry

end ratio_of_product_of_composites_l260_260271


namespace polynomial_expansion_identity_l260_260299

variable (a0 a1 a2 a3 a4 : ℝ)

theorem polynomial_expansion_identity
  (h : (2 - (x : ℝ))^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a0 - a1 + a2 - a3 + a4 = 81 :=
sorry

end polynomial_expansion_identity_l260_260299


namespace lightest_ball_box_is_blue_l260_260805

-- Define the weights and counts of balls
def yellow_ball_weight : ℕ := 50
def yellow_ball_count_per_box : ℕ := 50
def white_ball_weight : ℕ := 45
def white_ball_count_per_box : ℕ := 60
def blue_ball_weight : ℕ := 55
def blue_ball_count_per_box : ℕ := 40

-- Calculate the total weight of balls per type
def yellow_box_weight : ℕ := yellow_ball_weight * yellow_ball_count_per_box
def white_box_weight : ℕ := white_ball_weight * white_ball_count_per_box
def blue_box_weight : ℕ := blue_ball_weight * blue_ball_count_per_box

theorem lightest_ball_box_is_blue :
  (blue_box_weight < yellow_box_weight) ∧ (blue_box_weight < white_box_weight) :=
by
  -- Proof can go here
  sorry

end lightest_ball_box_is_blue_l260_260805


namespace intersection_points_and_verification_l260_260852

theorem intersection_points_and_verification :
  (∃ x y : ℝ, y = -3 * x ∧ y + 3 = 9 * x ∧ x = 1 / 4 ∧ y = -3 / 4) ∧
  ¬ (y = 2 * (1 / 4) - 1 ∧ (2 * (1 / 4) - 1 = -3 / 4)) :=
by
  sorry

end intersection_points_and_verification_l260_260852


namespace max_sum_cos_isosceles_triangle_l260_260379

theorem max_sum_cos_isosceles_triangle :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (2 * Real.cos α + Real.cos (π - 2 * α)) ≤ 1.5 :=
by
  sorry

end max_sum_cos_isosceles_triangle_l260_260379


namespace speed_rowing_upstream_l260_260992

theorem speed_rowing_upstream (V_m V_down : ℝ) (V_s V_up : ℝ)
  (h1 : V_m = 28) (h2 : V_down = 30) (h3 : V_down = V_m + V_s) (h4 : V_up = V_m - V_s) : 
  V_up = 26 :=
by
  sorry

end speed_rowing_upstream_l260_260992


namespace rick_ironed_27_pieces_l260_260784

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l260_260784


namespace original_classes_l260_260576

theorem original_classes (x : ℕ) (h1 : 280 % x = 0) (h2 : 585 % (x + 6) = 0) : x = 7 :=
sorry

end original_classes_l260_260576


namespace area_of_vegetable_patch_l260_260706

theorem area_of_vegetable_patch : ∃ (a b : ℕ), 
  (2 * (a + b) = 24 ∧ b = 3 * a + 2 ∧ (6 * (a + 1)) * (6 * (b + 1)) = 576) :=
sorry

end area_of_vegetable_patch_l260_260706


namespace cos_pi_plus_2alpha_value_l260_260720

theorem cos_pi_plus_2alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : 
    Real.cos (π + 2 * α) = 7 / 9 := sorry

end cos_pi_plus_2alpha_value_l260_260720


namespace henrys_distance_from_start_l260_260001

noncomputable def meters_to_feet (x : ℝ) : ℝ := x * 3.281
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem henrys_distance_from_start :
  let west_walk_feet := meters_to_feet 15
  let north_walk_feet := 60
  let east_walk_feet := 156
  let south_walk_meter_backwards := 30
  let south_walk_feet_backwards := 12
  let total_south_feet := meters_to_feet south_walk_meter_backwards + south_walk_feet_backwards
  let net_south_feet := total_south_feet - north_walk_feet
  let net_east_feet := east_walk_feet - west_walk_feet
  distance 0 0 net_east_feet (-net_south_feet) = 118 := 
by
  sorry

end henrys_distance_from_start_l260_260001


namespace car_avg_speed_l260_260960

def avg_speed_problem (d1 d2 t : ℕ) : ℕ :=
  (d1 + d2) / t

theorem car_avg_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 70) (h2 : d2 = 90) (ht : t = 2) :
  avg_speed_problem d1 d2 t = 80 := by
  sorry

end car_avg_speed_l260_260960


namespace combined_cost_increase_l260_260967

def original_bicycle_cost : ℝ := 200
def original_skates_cost : ℝ := 50
def bike_increase_percent : ℝ := 0.06
def skates_increase_percent : ℝ := 0.15

noncomputable def new_bicycle_cost : ℝ := original_bicycle_cost * (1 + bike_increase_percent)
noncomputable def new_skates_cost : ℝ := original_skates_cost * (1 + skates_increase_percent)
noncomputable def original_total_cost : ℝ := original_bicycle_cost + original_skates_cost
noncomputable def new_total_cost : ℝ := new_bicycle_cost + new_skates_cost
noncomputable def total_increase : ℝ := new_total_cost - original_total_cost
noncomputable def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem combined_cost_increase : percent_increase = 7.8 := by
  sorry

end combined_cost_increase_l260_260967


namespace Chang_solution_A_amount_l260_260757

def solution_alcohol_content (A B : ℝ) (x : ℝ) : ℝ :=
  0.16 * x + 0.10 * (x + 500)

theorem Chang_solution_A_amount (x : ℝ) :
  solution_alcohol_content 0.16 0.10 x = 76 → x = 100 :=
by
  intro h
  sorry

end Chang_solution_A_amount_l260_260757


namespace coordinates_of_F_double_prime_l260_260090

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem coordinates_of_F_double_prime :
  let F : ℝ × ℝ := (3, 3)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_x_axis F'
  F'' = (-3, -3) :=
by
  sorry

end coordinates_of_F_double_prime_l260_260090


namespace fox_appropriation_l260_260334

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l260_260334


namespace isosceles_triangle_angle_l260_260803

theorem isosceles_triangle_angle {x : ℝ} (hx0 : 0 < x) (hx1 : x < 90) (hx2 : 2 * x = 180 / 7) : x = 180 / 7 :=
sorry

end isosceles_triangle_angle_l260_260803


namespace total_tissues_brought_l260_260953

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l260_260953


namespace alpha_range_l260_260277

open Real

theorem alpha_range (α : ℝ) (h₀ : 0 ≤ α ∧ α ≤ π)
  (h₁ : ∀ x : ℝ, 8 * x^2 - 8 * sin α * x + cos (2 * α) ≥ 0) :
  (0 ≤ α ∧ α ≤ π / 6) ∨ (5 * π / 6 ≤ α ∧ α ≤ π) :=
by {
  sorry
}

end alpha_range_l260_260277


namespace evaluate_f_5_minus_f_neg_5_l260_260742

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end evaluate_f_5_minus_f_neg_5_l260_260742


namespace total_interest_rate_is_correct_l260_260684

theorem total_interest_rate_is_correct :
  let total_investment := 100000
  let interest_rate_first := 0.09
  let interest_rate_second := 0.11
  let invested_in_second := 29999.999999999993
  let invested_in_first := total_investment - invested_in_second
  let interest_first := invested_in_first * interest_rate_first
  let interest_second := invested_in_second * interest_rate_second
  let total_interest := interest_first + interest_second
  let total_interest_rate := (total_interest / total_investment) * 100
  total_interest_rate = 9.6 :=
by
  sorry

end total_interest_rate_is_correct_l260_260684


namespace gcd_of_6Tn2_and_nplus1_eq_2_l260_260546

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2_l260_260546


namespace exists_polynomial_distinct_powers_of_2_l260_260885

open Polynomial

variable (n : ℕ) (hn : n > 0)

theorem exists_polynomial_distinct_powers_of_2 :
  ∃ P : Polynomial ℤ, P.degree = n ∧ (∃ (k : Fin (n + 1) → ℕ), ∀ i j : Fin (n + 1), i ≠ j → 2 ^ k i ≠ 2 ^ k j ∧ (∀ i, P.eval i.val = 2 ^ k i)) :=
sorry

end exists_polynomial_distinct_powers_of_2_l260_260885


namespace option_C_not_like_terms_l260_260822

theorem option_C_not_like_terms :
  ¬ (2 * (m : ℝ) == 2 * (n : ℝ)) :=
by
  sorry

end option_C_not_like_terms_l260_260822


namespace part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l260_260987

-- Defining the conditions
def racket_price : ℕ := 50
def ball_price : ℕ := 20
def num_rackets : ℕ := 10

-- Store A cost function
def store_A_cost (x : ℕ) : ℕ := 20 * x + 300

-- Store B cost function
def store_B_cost (x : ℕ) : ℕ := 16 * x + 400

-- Part (1): Express the costs in algebraic form
theorem part1_store_a_cost (x : ℕ) (hx : 10 < x) : store_A_cost x = 20 * x + 300 := by
  sorry

theorem part1_store_b_cost (x : ℕ) (hx : 10 < x) : store_B_cost x = 16 * x + 400 := by
  sorry

-- Part (2): Cost for x = 40
theorem part2_cost_comparison : store_A_cost 40 > store_B_cost 40 := by
  sorry

-- Part (3): Most cost-effective purchasing plan
def store_a_cost_rackets : ℕ := racket_price * num_rackets
def store_a_free_balls : ℕ := num_rackets
def remaining_balls (total_balls : ℕ) : ℕ := total_balls - store_a_free_balls
def store_b_cost_remaining_balls (remaining_balls : ℕ) : ℕ := remaining_balls * ball_price * 4 / 5

theorem part3_cost_effective_plan : store_a_cost_rackets + store_b_cost_remaining_balls (remaining_balls 40) = 980 := by
  sorry

end part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l260_260987


namespace find_smallest_n_l260_260636

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l260_260636


namespace correct_option_l260_260512

-- Conditions
def option_A (a : ℕ) : Prop := (a^5)^2 = a^7
def option_B (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def option_C (a : ℕ) : Prop := (2 * a)^3 = 6 * a^3
def option_D (a : ℕ) : Prop := a^6 / a^2 = a^4

-- Theorem statement
theorem correct_option (a : ℕ) : ¬ option_A a ∧ ¬ option_B a ∧ ¬ option_C a ∧ option_D a := by
  sorry

end correct_option_l260_260512


namespace solution_set_of_inequality_l260_260949

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_f1_zero : f 1 = 0) : 
  { x | f x > 0 } = { x | x < -1 ∨ 1 < x } := 
by
  sorry

end solution_set_of_inequality_l260_260949


namespace percentage_decrease_l260_260478

variables (S : ℝ) (D : ℝ)
def initial_increase (S : ℝ) : ℝ := 1.5 * S
def final_gain (S : ℝ) : ℝ := 1.15 * S
def salary_after_decrease (S D : ℝ) : ℝ := (initial_increase S) * (1 - D)

theorem percentage_decrease :
  salary_after_decrease S D = final_gain S → D = 0.233333 :=
by
  sorry

end percentage_decrease_l260_260478


namespace smallest_positive_n_l260_260642

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260642


namespace blue_parrots_count_l260_260446

theorem blue_parrots_count (P : ℕ) (red green blue : ℕ) (h₁ : red = P / 2) (h₂ : green = P / 4) (h₃ : blue = P - red - green) (h₄ :  P + 30 = 150) : blue = 38 :=
by {
-- We will write the proof here
sorry
}

end blue_parrots_count_l260_260446


namespace men_left_hostel_l260_260241

variable (x : ℕ)
variable (h1 : 250 * 36 = (250 - x) * 45)

theorem men_left_hostel : x = 50 :=
by
  sorry

end men_left_hostel_l260_260241


namespace find_special_three_digit_numbers_l260_260869

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l260_260869


namespace rotated_point_coordinates_l260_260895

noncomputable def A : ℝ × ℝ := (1, 2)

def rotate_90_counterclockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, p.fst)

theorem rotated_point_coordinates :
  rotate_90_counterclockwise A = (-2, 1) :=
by
  -- Skipping the proof
  sorry

end rotated_point_coordinates_l260_260895


namespace gcd_4536_8721_l260_260877

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end gcd_4536_8721_l260_260877


namespace green_dots_third_row_l260_260050

noncomputable def row_difference (a b : Nat) : Nat := b - a

theorem green_dots_third_row (a1 a2 a4 a5 a3 d : Nat)
  (h_a1 : a1 = 3)
  (h_a2 : a2 = 6)
  (h_a4 : a4 = 12)
  (h_a5 : a5 = 15)
  (h_d : row_difference a2 a1 = d)
  (h_d_consistent : row_difference a2 a1 = row_difference a4 a3) :
  a3 = 9 :=
sorry

end green_dots_third_row_l260_260050


namespace sequence_formula_l260_260311

theorem sequence_formula (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 6)
  (h4 : a 4 = 10)
  (h5 : ∀ n > 0, a (n + 1) - a n = n + 1) :
  ∀ n, a n = n * (n + 1) / 2 :=
by 
  sorry

end sequence_formula_l260_260311


namespace initial_cupcakes_l260_260547

   theorem initial_cupcakes (X : ℕ) (condition : X - 20 + 20 = 26) : X = 26 :=
   by
     sorry
   
end initial_cupcakes_l260_260547


namespace complement_A_in_U_l260_260338

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} :=
by 
  sorry

end complement_A_in_U_l260_260338


namespace smallest_n_satisfies_conditions_l260_260626

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260626


namespace find_positive_integers_l260_260266

noncomputable def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem find_positive_integers (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hab_c : is_power_of_two (a * b - c))
  (hbc_a : is_power_of_two (b * c - a))
  (hca_b : is_power_of_two (c * a - b)) :
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) ∨
  (a = 2 ∧ b = 6 ∧ c = 11) :=
sorry

end find_positive_integers_l260_260266


namespace avg_scores_relation_l260_260695

variables (class_avg top8_avg other32_avg : ℝ)

theorem avg_scores_relation (h1 : 40 = 40) 
  (h2 : top8_avg = class_avg + 3) :
  other32_avg = top8_avg - 3.75 :=
sorry

end avg_scores_relation_l260_260695


namespace distinct_values_of_expression_l260_260902

variable {u v x y z : ℝ}

theorem distinct_values_of_expression (hu : u + u⁻¹ = x) (hv : v + v⁻¹ = y)
  (hx_distinct : x ≠ y) (hx_abs : |x| ≥ 2) (hy_abs : |y| ≥ 2) :
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z = u * v + (u * v)⁻¹)) →
  ∃ n, n = 2 := by 
    sorry

end distinct_values_of_expression_l260_260902


namespace negative_root_m_positive_l260_260450

noncomputable def is_negative_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 + m * x - 4 = 0

theorem negative_root_m_positive : ∀ m : ℝ, is_negative_root m → m > 0 :=
by
  intro m
  intro h
  sorry

end negative_root_m_positive_l260_260450


namespace simplify_trig_expression_l260_260481

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * Real.cos (α / 2) ^ 2) = 2 * Real.sin α :=
by
  sorry

end simplify_trig_expression_l260_260481


namespace automorphisms_and_anti_automorphisms_group_structure_l260_260466

variables (G : Type*) [Group G] [Nontrivial G] [Fintype G] [h_noncomm : ¬ (Commute : G → G → Prop)]

open Group

-- One-to-one mapping representing anti-automorphisms
def anti_automorphisms (f : G → G) : Prop := ∀ a b : G, f (a * b) = f b * f a

-- The main theorem statement
theorem automorphisms_and_anti_automorphisms_group_structure :
  let auto := Aut G in
  let antiauto := { f : G → G // anti_automorphisms G f } in
  ∃ (f : auto × (Zmod 2) ≃* (auto + antiauto)), true :=
begin
  -- Placeholder for proof
  sorry
end

end automorphisms_and_anti_automorphisms_group_structure_l260_260466


namespace box_office_scientific_notation_l260_260487

def billion : ℝ := 10^9
def box_office_revenue : ℝ := 57.44 * billion
def scientific_notation (n : ℝ) : ℝ × ℝ := (5.744, 10^10)

theorem box_office_scientific_notation :
  scientific_notation box_office_revenue = (5.744, 10^10) :=
by
  sorry

end box_office_scientific_notation_l260_260487


namespace smallest_possible_x2_plus_y2_l260_260916

theorem smallest_possible_x2_plus_y2 (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end smallest_possible_x2_plus_y2_l260_260916


namespace consecutive_odd_integers_sum_l260_260820

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 138) :
  x + (x + 2) + (x + 4) = 207 :=
sorry

end consecutive_odd_integers_sum_l260_260820


namespace binary_add_sub_l260_260254

theorem binary_add_sub:
  let a := 0b10110
  let b := 0b1010
  let c := 0b11100
  let d := 0b1110
  a + b - c + d = 0b01110 := by
  sorry

end binary_add_sub_l260_260254


namespace range_of_a_l260_260729

theorem range_of_a (p q : Prop)
  (hp : ∀ a : ℝ, (1 < a ↔ p))
  (hq : ∀ a : ℝ, (2 ≤ a ∨ a ≤ -2 ↔ q))
  (hpq : ∀ a : ℝ, ∀ (p : Prop), ∀ (q : Prop), (p ∧ q) → p ∧ q) :
    ∀ a : ℝ, p ∧ q → 2 ≤ a :=
sorry

end range_of_a_l260_260729


namespace common_number_in_sequences_l260_260686

theorem common_number_in_sequences (n m: ℕ) (a : ℕ)
    (h1 : a = 3 + 8 * n)
    (h2 : a = 5 + 9 * m)
    (h3 : 1 ≤ a ∧ a ≤ 200) : a = 131 :=
by
  sorry

end common_number_in_sequences_l260_260686


namespace petya_series_sum_l260_260191

theorem petya_series_sum (n k : ℕ) (h1 : (n + k) * (k + 1) = 20 * (n + 2 * k)) 
                                      (h2 : (n + k) * (k + 1) = 60 * n) :
  n = 29 ∧ k = 29 :=
by
  sorry

end petya_series_sum_l260_260191


namespace simplify_equation_l260_260578

theorem simplify_equation (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) -> 
  (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by 
  sorry

end simplify_equation_l260_260578


namespace find_value_of_M_l260_260083

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l260_260083


namespace number_of_ways_to_put_7_balls_in_2_boxes_l260_260564

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end number_of_ways_to_put_7_balls_in_2_boxes_l260_260564


namespace parallelogram_angle_solution_l260_260459

-- Define the geometrical setup
noncomputable def parallelogram (A B C D : Point) : Prop :=
  segment_parallel A D B C ∧ 
  segment_parallel A B D C ∧ 
  dist A B = dist B C ∧ 
  dist A D = dist D C

-- Given values
def sides (A B C D : Point) : Prop :=
  dist A B = 3 ∧
  dist A D = 5

-- Intersection point conditions
def intersection_points (A B C D M N P Q : Point) : Prop :=
  bisector_of_angle A M B ∧
  bisector_of_angle C N D ∧
  intersection C N D M P ∧
  intersection A M B N Q

-- Area condition
def area_condition (P Q : Point) : Prop :=
  parallelogram_area P Q = 6 / 5

-- Main theorem statement
theorem parallelogram_angle_solution (A B C D M N P Q : Point) 
  (h1 : parallelogram A B C D) 
  (h2 : sides A B C D) 
  (h3 : intersection_points A B C D M N P Q) 
  (h4 : area_condition P Q) : 
  ∃ (θ : ℝ), θ = Real.arcsin (1 / 3) ∨ θ = π - Real.arcsin (1 / 3) :=
sorry

end parallelogram_angle_solution_l260_260459


namespace range_of_m_l260_260550

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
by
  sorry

end range_of_m_l260_260550


namespace projection_is_orthocenter_l260_260573

-- Define a structure for a point in 3D space.
structure Point (α : Type) :=
(x : α)
(y : α)
(z : α)

-- Define mutually perpendicular edges condition.
def mutually_perpendicular {α : Type} [Field α] (A B C D : Point α) :=
(A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) + (A.z - D.z) * (B.z - D.z) = 0 ∧
(A.x - D.x) * (C.x - D.x) + (A.y - D.y) * (C.y - D.y) + (A.z - D.z) * (C.z - D.z) = 0 ∧
(B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) + (B.z - D.z) * (C.z - D.z) = 0

-- The main theorem statement.
theorem projection_is_orthocenter {α : Type} [Field α]
    (A B C D : Point α) (h : mutually_perpendicular A B C D) :
    ∃ O : Point α, -- there exists a point O (the orthocenter)
    (O.x * (B.y - A.y) + O.y * (A.y - B.y) + O.z * (A.y - B.y)) = 0 ∧
    (O.x * (C.y - B.y) + O.y * (B.y - C.y) + O.z * (B.y - C.y)) = 0 ∧
    (O.x * (A.y - C.y) + O.y * (C.y - A.y) + O.z * (C.y - A.y)) = 0 := 
sorry

end projection_is_orthocenter_l260_260573


namespace intersection_is_1_l260_260560

def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {y | ∃ x ∈ M, y = x ^ 2}
theorem intersection_is_1 : M ∩ N = {1} := by
  sorry

end intersection_is_1_l260_260560


namespace maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l260_260409

def pine_tree_height : ℚ := 37 / 4  -- 9 1/4 feet
def maple_tree_height : ℚ := 62 / 4  -- 15 1/2 feet (converted directly to common denominator)
def growth_rate : ℚ := 7 / 4  -- 1 3/4 feet per year

theorem maple_tree_taller_than_pine_tree : maple_tree_height - pine_tree_height = 25 / 4 := 
by sorry

theorem pine_tree_height_in_one_year : pine_tree_height + growth_rate = 44 / 4 := 
by sorry

end maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l260_260409


namespace find_number_l260_260771

-- Define the condition given in the problem
def condition (x : ℤ) := 13 * x - 272 = 105

-- Prove that given the condition, x equals 29
theorem find_number : ∃ x : ℤ, condition x ∧ x = 29 :=
by
  use 29
  unfold condition
  sorry

end find_number_l260_260771


namespace high_school_students_total_l260_260103

theorem high_school_students_total
    (students_taking_music : ℕ)
    (students_taking_art : ℕ)
    (students_taking_both_music_and_art : ℕ)
    (students_taking_neither : ℕ)
    (h1 : students_taking_music = 50)
    (h2 : students_taking_art = 20)
    (h3 : students_taking_both_music_and_art = 10)
    (h4 : students_taking_neither = 440) :
    students_taking_music - students_taking_both_music_and_art + students_taking_art - students_taking_both_music_and_art + students_taking_both_music_and_art + students_taking_neither = 500 :=
by
  sorry

end high_school_students_total_l260_260103


namespace range_of_h_l260_260951

theorem range_of_h 
  (y1 y2 y3 k : ℝ)
  (h : ℝ)
  (H1 : y1 = (-3 - h)^2 + k)
  (H2 : y2 = (-1 - h)^2 + k)
  (H3 : y3 = (1 - h)^2 + k)
  (H_ord : y2 < y1 ∧ y1 < y3) : 
  -2 < h ∧ h < -1 :=
sorry

end range_of_h_l260_260951


namespace part_i_part_ii_l260_260677

noncomputable def a_n (n : ℕ) : ℝ := 2^(n+2)
def b_n (n : ℕ) : ℝ := 1 / (n * Real.log2 (a_n n))
def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, (b_n (i+1))

theorem part_i (n : ℕ) : a_n n = 2^(n+2) :=
by sorry

theorem part_ii (n : ℕ) : S_n n = (3/4) - ((2*n+3) / (2 * (n+1) * (n+2))) :=
by sorry

end part_i_part_ii_l260_260677


namespace geometric_sequence_problem_l260_260470

-- Step d) Rewrite the problem in Lean 4 statement
theorem geometric_sequence_problem 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (q : ℝ) 
  (h1 : ∀ n, n > 0 → a_n n = 1 * q^(n-1)) 
  (h2 : 1 + q + q^2 = 7)
  (h3 : 6 * 1 * q = 1 + 3 + 1 * q^2 + 4)
  :
  (∀ n, a_n n = 2^(n-1)) ∧ 
  (∀ n, T_n n = 4 - (n+2) / 2^(n-1)) :=
  sorry

end geometric_sequence_problem_l260_260470


namespace exists_polynomial_divisible_by_power_of_x_minus_one_l260_260351

theorem exists_polynomial_divisible_by_power_of_x_minus_one (n : ℕ) :
  ∃ P : Polynomial ℤ,
    (∀ k : ℕ, P.coeff k ∈ {0, -1, 1}) ∧
    P.degree ≤ (2^n - 1) ∧
    (Polynomial.X - 1) ^ n ∣ P :=
by
  sorry

end exists_polynomial_divisible_by_power_of_x_minus_one_l260_260351


namespace emily_beads_l260_260410

-- Definitions of the conditions as per step a)
def beads_per_necklace : ℕ := 8
def necklaces : ℕ := 2

-- Theorem statement to prove the equivalent math problem
theorem emily_beads : beads_per_necklace * necklaces = 16 :=
by
  sorry

end emily_beads_l260_260410


namespace class_trip_contributions_l260_260377

theorem class_trip_contributions (x y : ℕ) :
  (x + 5) * (y + 6) = x * y + 792 ∧ (x - 4) * (y + 4) = x * y - 388 → x = 213 ∧ y = 120 := 
by
  sorry

end class_trip_contributions_l260_260377


namespace solution_set_of_inequality_l260_260492

theorem solution_set_of_inequality (x : ℝ) : (x^2 - 2*x - 5 > 2*x) ↔ (x > 5 ∨ x < -1) :=
by sorry

end solution_set_of_inequality_l260_260492


namespace problem_statement_l260_260187

def g (x : ℕ) : ℕ := x^2 - 4 * x

theorem problem_statement :
  g (g (g (g (g (g 2))))) = L := sorry

end problem_statement_l260_260187


namespace find_probability_l260_260021

open Probability

noncomputable def X : NormalDist := NormalDist.mk 3 1

theorem find_probability : 
  P(0 < fun x : ℝ => X.density x ∘ 1) = 0.0215 :=
by 
  sorry

end find_probability_l260_260021


namespace find_three_digit_numbers_l260_260862

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l260_260862


namespace joe_height_l260_260064

theorem joe_height (S J A : ℝ) (h1 : S + J + A = 180) (h2 : J = 2 * S + 6) (h3 : A = S - 3) : J = 94.5 :=
by 
  -- Lean proof goes here
  sorry

end joe_height_l260_260064


namespace deck_cost_l260_260218

variable (rareCount : ℕ := 19)
variable (uncommonCount : ℕ := 11)
variable (commonCount : ℕ := 30)
variable (rareCost : ℝ := 1.0)
variable (uncommonCost : ℝ := 0.5)
variable (commonCost : ℝ := 0.25)

theorem deck_cost : rareCount * rareCost + uncommonCount * uncommonCost + commonCount * commonCost = 32 := by
  sorry

end deck_cost_l260_260218


namespace mix_ratios_l260_260098

theorem mix_ratios (milk1 water1 milk2 water2 : ℕ) 
  (h1 : milk1 = 7) (h2 : water1 = 2)
  (h3 : milk2 = 8) (h4 : water2 = 1) :
  (milk1 + milk2) / (water1 + water2) = 5 :=
by
  -- Proof required here
  sorry

end mix_ratios_l260_260098


namespace train_crossing_time_l260_260761

theorem train_crossing_time
  (train_length : ℕ)           -- length of the train in meters
  (train_speed_kmh : ℕ)        -- speed of the train in kilometers per hour
  (conversion_factor : ℕ)      -- conversion factor from km/hr to m/s
  (train_speed_ms : ℕ)         -- speed of the train in meters per second
  (time_to_cross : ℚ)          -- time to cross in seconds
  (h1 : train_length = 60)
  (h2 : train_speed_kmh = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : train_speed_ms = train_speed_kmh * conversion_factor)
  (h5 : time_to_cross = train_length / train_speed_ms) :
  time_to_cross = 1.5 :=
by sorry

end train_crossing_time_l260_260761


namespace restaurant_A2_probability_l260_260200

noncomputable def prob_A2 (P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 : ℝ) : ℝ :=
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1

theorem restaurant_A2_probability :
  let P_A1 := 0.4
  let P_B1 := 0.6
  let P_A2_given_A1 := 0.6
  let P_A2_given_B1 := 0.5
  prob_A2 P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 = 0.54 :=
by
  sorry

end restaurant_A2_probability_l260_260200


namespace find_larger_number_l260_260304

theorem find_larger_number (x y : ℤ) (h1 : 5 * y = 6 * x) (h2 : y - x = 12) : y = 72 :=
sorry

end find_larger_number_l260_260304


namespace smallest_n_45_l260_260633

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k * k

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m * m

theorem smallest_n_45 :
  ∃ n : ℕ, n > 0 ∧ (is_perfect_square (5 * n)) ∧ (is_perfect_cube (3 * n)) ∧ ∀ m : ℕ, (m > 0 ∧ (is_perfect_square (5 * m)) ∧ (is_perfect_cube (3 * m))) → n ≤ m :=
sorry

end smallest_n_45_l260_260633


namespace plate_arrangement_l260_260391

def arrangements_without_restriction : Nat :=
  Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 3)

def arrangements_adjacent_green : Nat :=
  (Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3)) * Nat.factorial 3

def allowed_arrangements : Nat :=
  arrangements_without_restriction - arrangements_adjacent_green

theorem plate_arrangement : 
  allowed_arrangements = 2520 := 
by
  sorry

end plate_arrangement_l260_260391


namespace range_of_m_l260_260738

theorem range_of_m (m x : ℝ) (h : (x + m) / 3 - (2 * x - 1) / 2 = m) (hx : x ≤ 0) : m ≥ 3 / 4 := 
sorry

end range_of_m_l260_260738


namespace length_PQ_l260_260285

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

def P : Point3D := { x := 3, y := 4, z := 5 }

def Q : Point3D := { x := 3, y := 4, z := 0 }

theorem length_PQ : distance P Q = 5 :=
by
  sorry

end length_PQ_l260_260285


namespace base_4_digits_l260_260979

theorem base_4_digits (b : ℕ) (h1 : b^3 ≤ 216) (h2 : 216 < b^4) : b = 5 :=
sorry

end base_4_digits_l260_260979


namespace dividend_50100_l260_260454

theorem dividend_50100 (D Q R : ℕ) (h1 : D = 20 * Q) (h2 : D = 10 * R) (h3 : R = 100) : 
    D * Q + R = 50100 := by
  sorry

end dividend_50100_l260_260454


namespace kolya_pays_90_rubles_l260_260252

theorem kolya_pays_90_rubles {x y : ℝ} 
  (h1 : x + 3 * y = 78) 
  (h2 : x + 8 * y = 108) :
  x + 5 * y = 90 :=
by sorry

end kolya_pays_90_rubles_l260_260252


namespace loss_calculation_l260_260802

-- Given conditions: 
-- The ratio of the amount of money Cara, Janet, and Jerry have is 4:5:6
-- The total amount of money they have is $75

theorem loss_calculation :
  let cara_ratio := 4
  let janet_ratio := 5
  let jerry_ratio := 6
  let total_ratio := cara_ratio + janet_ratio + jerry_ratio
  let total_money := 75
  let part_value := total_money / total_ratio
  let cara_money := cara_ratio * part_value
  let janet_money := janet_ratio * part_value
  let combined_money := cara_money + janet_money
  let selling_price := 0.80 * combined_money
  combined_money - selling_price = 9 :=
by
  sorry

end loss_calculation_l260_260802


namespace charlie_and_elle_crayons_l260_260184

theorem charlie_and_elle_crayons :
  (∃ (Lizzie Bobbie Billie Charlie Dave Elle : ℕ),
  Billie = 18 ∧
  Bobbie = 3 * Billie ∧
  Lizzie = Bobbie / 2 ∧
  Charlie = 2 * Lizzie ∧
  Dave = 4 * Billie ∧
  Elle = (Bobbie + Dave) / 2 ∧
  Charlie + Elle = 117) :=
sorry

end charlie_and_elle_crayons_l260_260184


namespace fox_cub_distribution_l260_260332

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l260_260332


namespace parabola_vertex_and_point_l260_260848

theorem parabola_vertex_and_point (a b c : ℝ) : 
  (∀ x, y = a * x^2 + b * x + c) ∧ 
  ∃ x y, (y = a * (x - 4)^2 + 3) → 
  (a * 2^2 + b * 2 + c = 5) → 
  (a = 1/2 ∧ b = -4 ∧ c = 11) :=
by
  sorry

end parabola_vertex_and_point_l260_260848


namespace number_of_cupcakes_l260_260527

theorem number_of_cupcakes (total gluten_free vegan gluten_free_vegan non_vegan : ℕ) 
    (h1 : gluten_free = total / 2)
    (h2 : vegan = 24)
    (h3 : gluten_free_vegan = vegan / 2)
    (h4 : non_vegan = 28)
    (h5 : gluten_free_vegan = gluten_free / 2) :
    total = 52 :=
by
  sorry

end number_of_cupcakes_l260_260527


namespace min_value_of_x_plus_y_l260_260890

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l260_260890


namespace equation_solution_1_equation_solution_2_equation_solution_3_l260_260790

def system_of_equations (x y : ℝ) : Prop :=
  (x * (x^2 - 3 * y^2) = 16) ∧ (y * (3 * x^2 - y^2) = 88)

theorem equation_solution_1 :
  system_of_equations 4 2 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_2 :
  system_of_equations (-3.7) 2.5 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_3 :
  system_of_equations (-0.3) (-4.5) :=
by
  -- The proof is skipped.
  sorry

end equation_solution_1_equation_solution_2_equation_solution_3_l260_260790


namespace sum_of_distinct_prime_factors_l260_260713

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end sum_of_distinct_prime_factors_l260_260713


namespace arithmetic_sequence_a5_l260_260028

theorem arithmetic_sequence_a5 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : S 4 = 16)
  (h_sum : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h_a : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  a 5 = 9 :=
by 
  sorry

end arithmetic_sequence_a5_l260_260028


namespace number_of_ways_to_distribute_balls_l260_260568

theorem number_of_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 7 → boxes = 2 → 
  (∑ i in finset.range (balls + 1), nat.choose balls i / (if i == balls / 2 then 1 else 2)) = 64 :=
by
  intros balls boxes h1 h2
  sorry

end number_of_ways_to_distribute_balls_l260_260568


namespace lines_parallel_if_perpendicular_to_same_plane_l260_260444

variables {Line : Type} {Plane : Type}
variable (a b : Line)
variable (α : Plane)

-- Conditions 
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- Definition for line perpendicular to plane
def lines_parallel (l1 l2 : Line) : Prop := sorry -- Definition for lines parallel

-- Theorem Statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  line_perpendicular_to_plane a α →
  line_perpendicular_to_plane b α →
  lines_parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l260_260444


namespace curve_symmetric_origin_l260_260211

theorem curve_symmetric_origin (x y : ℝ) (h : 3*x^2 - 8*x*y + 2*y^2 = 0) :
  3*(-x)^2 - 8*(-x)*(-y) + 2*(-y)^2 = 3*x^2 - 8*x*y + 2*y^2 :=
sorry

end curve_symmetric_origin_l260_260211


namespace relationship_of_points_on_inverse_proportion_l260_260772

theorem relationship_of_points_on_inverse_proportion :
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  sorry

end relationship_of_points_on_inverse_proportion_l260_260772


namespace smallest_possible_b_l260_260183

-- Definition of the polynomial Q(x)
def Q (x : ℤ) : ℤ := sorry -- Polynomial with integer coefficients

-- Initial conditions for b and Q
variable (b : ℤ) (hb : b > 0)
variable (hQ1 : Q 2 = b)
variable (hQ2 : Q 4 = b)
variable (hQ3 : Q 6 = b)
variable (hQ4 : Q 8 = b)
variable (hQ5 : Q 1 = -b)
variable (hQ6 : Q 3 = -b)
variable (hQ7 : Q 5 = -b)
variable (hQ8 : Q 7 = -b)

theorem smallest_possible_b : b = 315 :=
by
  sorry

end smallest_possible_b_l260_260183


namespace find_a_value_l260_260273

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a_value :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a 0 + f a 1 = a) → a = 1 / 2 :=
by
  sorry

end find_a_value_l260_260273


namespace A_share_of_profit_l260_260234

-- Define necessary financial terms and operations
def initial_investment_A := 3000
def initial_investment_B := 4000

def withdrawal_A := 1000
def advanced_B := 1000

def duration_initial := 8
def duration_remaining := 4

def total_profit := 630

-- Calculate the equivalent investment duration for A and B
def investment_months_A_first := initial_investment_A * duration_initial
def investment_months_A_remaining := (initial_investment_A - withdrawal_A) * duration_remaining
def investment_months_A := investment_months_A_first + investment_months_A_remaining

def investment_months_B_first := initial_investment_B * duration_initial
def investment_months_B_remaining := (initial_investment_B + advanced_B) * duration_remaining
def investment_months_B := investment_months_B_first + investment_months_B_remaining

-- Prove that A's share of the profit is Rs. 240
theorem A_share_of_profit : 
  let ratio_A : ℚ := 4
  let ratio_B : ℚ := 6.5
  let total_ratio : ℚ := ratio_A + ratio_B
  let a_share : ℚ := (total_profit * ratio_A) / total_ratio
  a_share = 240 := 
by
  sorry

end A_share_of_profit_l260_260234


namespace sum_of_six_terms_l260_260894

theorem sum_of_six_terms (a1 : ℝ) (S4 : ℝ) (d : ℝ) (a1_eq : a1 = 1 / 2) (S4_eq : S4 = 20) :
  S4 = (4 * a1 + (4 * (4 - 1) / 2) * d) → (S4 = 20) →
  (6 * a1 + (6 * (6 - 1) / 2) * d = 48) :=
by
  intros
  sorry

end sum_of_six_terms_l260_260894


namespace trevor_comic_first_issue_pages_l260_260504

theorem trevor_comic_first_issue_pages
  (x : ℕ) 
  (h1 : 3 * x + 4 = 220) :
  x = 72 := 
by
  sorry

end trevor_comic_first_issue_pages_l260_260504


namespace average_salary_correct_l260_260208

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct_l260_260208


namespace range_of_a_l260_260291

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x
noncomputable def g (x a : ℝ) : ℝ := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, x1 ∈ Set.Icc 0 2 → ∃ x2 : ℝ, x2 ∈ Set.Ioi a ∧ f x1 ≥ g x2 a) →
  a ≤ -1 :=
by
  intro h
  sorry

end range_of_a_l260_260291


namespace total_cost_of_dresses_l260_260015

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l260_260015


namespace charge_move_increases_energy_l260_260360

noncomputable def energy_increase_when_charge_moved : ℝ :=
  let initial_energy := 15
  let energy_per_pair := initial_energy / 3
  let new_energy_AB := energy_per_pair
  let new_energy_AC := 2 * energy_per_pair
  let new_energy_BC := 2 * energy_per_pair
  let final_energy := new_energy_AB + new_energy_AC + new_energy_BC
  final_energy - initial_energy

theorem charge_move_increases_energy :
  energy_increase_when_charge_moved = 10 :=
by
  sorry

end charge_move_increases_energy_l260_260360


namespace set_union_inter_eq_l260_260442

open Set

-- Conditions: Definitions of sets M, N, and P
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3, 4, 5}

-- Claim: The result of (M ∩ N) ∪ P equals {1, 2, 3, 4, 5}
theorem set_union_inter_eq :
  (M ∩ N ∪ P) = {1, 2, 3, 4, 5} := 
by
  sorry

end set_union_inter_eq_l260_260442


namespace ethanol_solution_exists_l260_260513

noncomputable def ethanol_problem : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 204 ∧ 0.12 * x + 0.16 * (204 - x) = 30

theorem ethanol_solution_exists : ethanol_problem :=
sorry

end ethanol_solution_exists_l260_260513


namespace exists_n_with_common_divisor_l260_260268

theorem exists_n_with_common_divisor :
  ∃ (n : ℕ), ∀ (k : ℕ), (k ≤ 20) → Nat.gcd (n + k) 30030 > 1 :=
by
  sorry

end exists_n_with_common_divisor_l260_260268


namespace largest_lcm_value_l260_260378

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 := by
sorry

end largest_lcm_value_l260_260378


namespace find_uncommon_cards_l260_260813

def numRare : ℕ := 19
def numCommon : ℕ := 30
def costRare : ℝ := 1
def costUncommon : ℝ := 0.50
def costCommon : ℝ := 0.25
def totalCostDeck : ℝ := 32

theorem find_uncommon_cards (U : ℕ) (h : U * costUncommon + numRare * costRare + numCommon * costCommon = totalCostDeck) : U = 11 := by
  sorry

end find_uncommon_cards_l260_260813


namespace part1_part2_l260_260117

-- Define the constants based on given conditions
def cost_price : ℕ := 5
def initial_selling_price : ℕ := 9
def initial_sales_volume : ℕ := 32
def price_increment : ℕ := 2
def sales_decrement : ℕ := 8

-- Part 1: Define the elements 
def selling_price_part1 : ℕ := 11
def profit_per_item_part1 : ℕ := 6
def daily_sales_volume_part1 : ℕ := 24

theorem part1 :
  (selling_price_part1 - cost_price = profit_per_item_part1) ∧ 
  (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price_part1 - initial_selling_price) = daily_sales_volume_part1) := 
by
  sorry

-- Part 2: Define the elements 
def target_daily_profit : ℕ := 140
def selling_price1_part2 : ℕ := 12
def selling_price2_part2 : ℕ := 10

theorem part2 :
  (((selling_price1_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price1_part2 - initial_selling_price)) = target_daily_profit) ∨
  ((selling_price2_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price2_part2 - initial_selling_price)) = target_daily_profit)) :=
by
  sorry

end part1_part2_l260_260117


namespace cost_per_bracelet_l260_260942

/-- Each friend and the number of their name's letters -/
def friends_letters_counts : List (String × Nat) :=
  [("Jessica", 7), ("Tori", 4), ("Lily", 4), ("Patrice", 7)]

/-- Total cost spent by Robin -/
def total_cost : Nat := 44

/-- Calculate the total number of bracelets -/
def total_bracelets : Nat :=
  friends_letters_counts.foldr (λ p acc => p.snd + acc) 0

theorem cost_per_bracelet : (total_cost / total_bracelets) = 2 :=
  by
    sorry

end cost_per_bracelet_l260_260942


namespace factorization_roots_l260_260425

theorem factorization_roots (x : ℂ) : 
  (x^3 - 2*x^2 - x + 2) * (x - 3) * (x + 1) = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  -- Note: Proof to be completed
  sorry

end factorization_roots_l260_260425


namespace find_x_of_equation_l260_260423

theorem find_x_of_equation (x : ℝ) (hx : x ≠ 0) : (7 * x)^4 = (14 * x)^3 → x = 8 / 7 :=
by
  intro h
  sorry

end find_x_of_equation_l260_260423


namespace geometric_sequence_problem_l260_260579

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem_l260_260579


namespace kevin_sold_13_crates_of_grapes_l260_260172

-- Define the conditions
def total_crates : ℕ := 50
def crates_of_mangoes : ℕ := 20
def crates_of_passion_fruits : ℕ := 17

-- Define the question and expected answer
def crates_of_grapes : ℕ := total_crates - (crates_of_mangoes + crates_of_passion_fruits)

-- Prove that the crates of grapes equals to 13
theorem kevin_sold_13_crates_of_grapes :
  crates_of_grapes = 13 :=
by
  -- The proof steps are omitted as per instructions
  sorry

end kevin_sold_13_crates_of_grapes_l260_260172


namespace remaining_credit_l260_260590

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l260_260590


namespace ellipse_hexagon_proof_l260_260899

noncomputable def m_value : ℝ := 3 + 2 * Real.sqrt 3

theorem ellipse_hexagon_proof (m : ℝ) (k : ℝ) 
  (hk : k ≠ 0) (hm : m > 3) :
  (∀ x y : ℝ, (x / m)^2 + (y / 3)^2 = 1 ∧ (y = k * x ∨ y = -k * x)) →
  k = Real.sqrt 3 →
  (|((4*m)/(m+1)) - (m-3)| = 0) →
  m = m_value :=
by
  sorry

end ellipse_hexagon_proof_l260_260899


namespace great_white_shark_teeth_is_420_l260_260998

-- Define the number of teeth in a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Define the number of teeth in a hammerhead shark based on the tiger shark's teeth
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Define the number of teeth in a great white shark based on the sum of tiger and hammerhead shark's teeth
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- The theorem statement that we need to prove
theorem great_white_shark_teeth_is_420 : great_white_shark_teeth = 420 :=
by
  -- Provide space for the proof
  sorry

end great_white_shark_teeth_is_420_l260_260998


namespace abc_divisibility_l260_260265

theorem abc_divisibility (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) : 
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by {
  sorry  -- proof to be filled in
}

end abc_divisibility_l260_260265


namespace total_flour_l260_260193

def bought_rye_flour := 5
def bought_bread_flour := 10
def bought_chickpea_flour := 3
def had_pantry_flour := 2

theorem total_flour : bought_rye_flour + bought_bread_flour + bought_chickpea_flour + had_pantry_flour = 20 :=
by
  sorry

end total_flour_l260_260193


namespace harold_shared_with_five_friends_l260_260562

theorem harold_shared_with_five_friends 
  (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) (shared : ℕ) (friends : ℕ)
  (H1 : total_marbles = 100)
  (H2 : kept_marbles = 20)
  (H3 : marbles_per_friend = 16)
  (H4 : shared = total_marbles - kept_marbles)
  (H5 : friends = shared / marbles_per_friend) :
  friends = 5 :=
by
  sorry

end harold_shared_with_five_friends_l260_260562


namespace domain_of_g_l260_260897

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

theorem domain_of_g :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)} = -- Expected domain of g(x)
  { x : ℝ |
    (0 ≤ x ∧ x ≤ 6) ∧ -- Domain of f is 0 ≤ x ≤ 6
    2 * x ≤ 6 ∧ -- For g(x) to be in the domain of f(2x)
    0 ≤ 2 * x ∧ -- Ensures 2x fits within the domain 0 < 2x < 6
    x ≠ 2 } -- x cannot be 2
:= sorry

end domain_of_g_l260_260897


namespace negate_exists_implies_forall_l260_260077

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l260_260077


namespace distance_to_right_focus_l260_260282

variable (F1 F2 P : ℝ × ℝ)
variable (a : ℝ)
variable (h_ellipse : ∀ P : ℝ × ℝ, P ∈ { P : ℝ × ℝ | (P.1^2 / 9) + (P.2^2 / 8) = 1 })
variable (h_foci_dist : (P : ℝ × ℝ) → (F1 : ℝ × ℝ) → (F2 : ℝ × ℝ) → (dist P F1) = 2)
variable (semi_major_axis : a = 3)

theorem distance_to_right_focus (h : dist F1 F2 = 2 * a) : dist P F2 = 4 := 
sorry

end distance_to_right_focus_l260_260282


namespace multiply_negatives_l260_260086

theorem multiply_negatives : (-3) * (-4) * (-1) = -12 := 
by sorry

end multiply_negatives_l260_260086


namespace find_positive_value_of_X_l260_260042

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l260_260042


namespace sum_of_ages_l260_260991

variable (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) (A4 : ℝ) (A5 : ℝ) (A6 : ℝ) (A7 : ℝ)

noncomputable def age_first_scroll := 4080
noncomputable def age_difference := 2040

theorem sum_of_ages :
  let r := (age_difference:ℝ) / (age_first_scroll:ℝ)
  let A2 := (age_first_scroll:ℝ) + age_difference
  let A3 := A2 + (A2 - age_first_scroll) * r
  let A4 := A3 + (A3 - A2) * r
  let A5 := A4 + (A4 - A3) * r
  let A6 := A5 + (A5 - A4) * r
  let A7 := A6 + (A6 - A5) * r
  (age_first_scroll:ℝ) + A2 + A3 + A4 + A5 + A6 + A7 = 41023.75 := 
  by sorry

end sum_of_ages_l260_260991


namespace quadratic_inequality_solution_l260_260370

theorem quadratic_inequality_solution : 
  ∀ x : ℝ, (2 * x ^ 2 + 7 * x + 3 > 0) ↔ (x < -3 ∨ x > -0.5) :=
by
  sorry

end quadratic_inequality_solution_l260_260370


namespace area_difference_l260_260490

-- Setting up the relevant conditions and entities
def side_red := 8
def length_yellow := 10
def width_yellow := 5

-- Definition of areas
def area_red := side_red * side_red
def area_yellow := length_yellow * width_yellow

-- The theorem we need to prove
theorem area_difference :
  area_red - area_yellow = 14 :=
by
  -- We skip the proof here due to the instruction
  sorry

end area_difference_l260_260490


namespace original_denominator_l260_260833

theorem original_denominator (d : ℕ) (h : 11 = 3 * (d + 8)) : d = 25 :=
by
  sorry

end original_denominator_l260_260833


namespace total_clothing_ironed_l260_260786

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l260_260786


namespace smallest_n_satisfies_conditions_l260_260650

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end smallest_n_satisfies_conditions_l260_260650


namespace proof_problem_l260_260033

noncomputable def problem_equivalent_proof (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧
  (z + 6 = 2 * y - z) ∧
  (x + 8 * z = y + 2) →
  (x^2 + y^2 + z^2 = 21)

theorem proof_problem (x y z : ℝ) : problem_equivalent_proof x y z :=
by
  sorry

end proof_problem_l260_260033


namespace prism_volume_l260_260224

theorem prism_volume (x y z : ℝ) (h1 : x * y = 24) (h2 : y * z = 8) (h3 : x * z = 3) : 
  x * y * z = 24 :=
sorry

end prism_volume_l260_260224


namespace chocolate_bar_cost_l260_260704

theorem chocolate_bar_cost (x : ℝ) (total_bars : ℕ) (bars_sold : ℕ) (total_amount_made : ℝ)
    (h1 : total_bars = 7)
    (h2 : bars_sold = total_bars - 4)
    (h3 : total_amount_made = 9)
    (h4 : total_amount_made = bars_sold * x) : x = 3 :=
sorry

end chocolate_bar_cost_l260_260704


namespace digits_count_of_special_numbers_l260_260807

theorem digits_count_of_special_numbers
  (n : ℕ)
  (h1 : 8^n = 28672) : n = 5 := 
by
  sorry

end digits_count_of_special_numbers_l260_260807


namespace probability_point_on_hyperbola_l260_260063

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l260_260063


namespace total_amount_spent_l260_260017

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l260_260017


namespace trig_problem_l260_260160

variables (θ : ℝ)

theorem trig_problem (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.tan θ + 1 / Real.tan θ = 4 :=
sorry

end trig_problem_l260_260160


namespace no_positive_integer_k_for_rational_solutions_l260_260408

theorem no_positive_integer_k_for_rational_solutions :
  ∀ k : ℕ, k > 0 → ¬ ∃ m : ℤ, 12 * (27 - k ^ 2) = m ^ 2 := by
  sorry

end no_positive_integer_k_for_rational_solutions_l260_260408


namespace point_on_hyperbola_probability_l260_260056

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l260_260056


namespace speed_of_man_cycling_l260_260616

theorem speed_of_man_cycling (L B : ℝ) (h1 : L / B = 1 / 3) (h2 : B = 3 * L)
  (h3 : L * B = 30000) (h4 : ∀ t : ℝ, t = 4 / 60): 
  ( (2 * L + 2 * B) / (4 / 60) ) = 12000 :=
by
  -- Assume given conditions
  sorry

end speed_of_man_cycling_l260_260616


namespace striped_turtles_adult_percentage_l260_260812

noncomputable def percentage_of_adult_striped_turtles (total_turtles : ℕ) (female_percentage : ℝ) (stripes_per_male : ℕ) (baby_stripes : ℕ) : ℝ :=
  let total_male := total_turtles * (1 - female_percentage)
  let total_striped_male := total_male / stripes_per_male
  let adult_striped_males := total_striped_male - baby_stripes
  (adult_striped_males / total_striped_male) * 100

theorem striped_turtles_adult_percentage :
  percentage_of_adult_striped_turtles 100 0.60 4 4 = 60 := 
  by
  -- proof omitted
  sorry

end striped_turtles_adult_percentage_l260_260812


namespace cone_volume_ratio_l260_260830

noncomputable def ratio_of_volumes (r h : ℝ) : ℝ :=
  let S1 := r^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 12
  let S2 := r^2 * (10 * Real.pi + 3 * Real.sqrt 3) / 12
  S1 / S2

theorem cone_volume_ratio (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  ratio_of_volumes r h = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3) :=
  sorry

end cone_volume_ratio_l260_260830


namespace time_to_cross_is_correct_l260_260835

noncomputable def train_cross_bridge_time : ℝ :=
  let length_train := 130
  let speed_train_kmh := 45
  let length_bridge := 245.03
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_ms
  time

theorem time_to_cross_is_correct : train_cross_bridge_time = 30.0024 :=
by
  sorry

end time_to_cross_is_correct_l260_260835


namespace fifteenth_triangular_number_is_120_l260_260700

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem fifteenth_triangular_number_is_120 : triangular_number 15 = 120 := by
  sorry

end fifteenth_triangular_number_is_120_l260_260700


namespace ann_frosting_time_l260_260934

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l260_260934


namespace quadratic_roots_range_l260_260900

variable (a : ℝ)

theorem quadratic_roots_range (h : ∀ b c (eq : b = -a ∧ c = a^2 - 4), ∃ x y, x ≠ y ∧ x^2 + b * x + c = 0 ∧ x > 0 ∧ y^2 + b * y + c = 0) :
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end quadratic_roots_range_l260_260900


namespace bedroom_light_energy_usage_l260_260349

-- Define the conditions and constants
def noahs_bedroom_light_usage (W : ℕ) : ℕ := W
def noahs_office_light_usage (W : ℕ) : ℕ := 3 * W
def noahs_living_room_light_usage (W : ℕ) : ℕ := 4 * W
def total_energy_used (W : ℕ) : ℕ := 2 * (noahs_bedroom_light_usage W + noahs_office_light_usage W + noahs_living_room_light_usage W)
def energy_consumption := 96

-- The main theorem to be proven
theorem bedroom_light_energy_usage : ∃ W : ℕ, total_energy_used W = energy_consumption ∧ W = 6 :=
by
  sorry

end bedroom_light_energy_usage_l260_260349


namespace moles_of_NH4Cl_combined_l260_260269

-- Define the chemical reaction equation
def reaction (NH4Cl H2O NH4OH HCl : ℕ) := 
  NH4Cl + H2O = NH4OH + HCl

-- Given conditions
def condition1 (H2O : ℕ) := H2O = 1
def condition2 (NH4OH : ℕ) := NH4OH = 1

-- Theorem statement: Prove that number of moles of NH4Cl combined is 1
theorem moles_of_NH4Cl_combined (H2O NH4OH NH4Cl HCl : ℕ) 
  (h1: condition1 H2O) (h2: condition2 NH4OH) (h3: reaction NH4Cl H2O NH4OH HCl) : 
  NH4Cl = 1 :=
sorry

end moles_of_NH4Cl_combined_l260_260269


namespace arithmetic_sequence_eleventh_term_l260_260127

theorem arithmetic_sequence_eleventh_term 
  (a d : ℚ)
  (h_sum_first_six : 6 * a + 15 * d = 30)
  (h_seventh_term : a + 6 * d = 10) : 
    a + 10 * d = 110 / 7 := 
by
  sorry

end arithmetic_sequence_eleventh_term_l260_260127


namespace quadratic_function_value_at_18_l260_260692

noncomputable def p (d e f x : ℝ) : ℝ := d*x^2 + e*x + f

theorem quadratic_function_value_at_18
  (d e f : ℝ)
  (h_sym : ∀ x1 x2 : ℝ, p d e f 6 = p d e f 12)
  (h_max : ∀ x : ℝ, x = 10 → ∃ p_max : ℝ, ∀ y : ℝ, p d e f x ≤ p_max)
  (h_p0 : p d e f 0 = -1) : 
  p d e f 18 = -1 := 
sorry

end quadratic_function_value_at_18_l260_260692


namespace eighth_L_prime_is_31_l260_260468

def setL := {n : ℕ | n > 0 ∧ n % 3 = 1}

def isLPrime (n : ℕ) : Prop :=
  n ∈ setL ∧ n ≠ 1 ∧ ∀ m ∈ setL, (m ∣ n) → (m = 1 ∨ m = n)

theorem eighth_L_prime_is_31 : 
  ∃ n ∈ setL, isLPrime n ∧ 
  (∀ k, (∃ m ∈ setL, isLPrime m ∧ m < n) → k < 8 → m ≠ n) :=
by sorry

end eighth_L_prime_is_31_l260_260468


namespace intersection_A_B_l260_260475

def setA (x : ℝ) : Prop := x^2 - 2 * x > 0
def setB (x : ℝ) : Prop := abs (x + 1) < 2

theorem intersection_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -3 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l260_260475


namespace mary_needs_10_charges_to_vacuum_house_l260_260947

theorem mary_needs_10_charges_to_vacuum_house :
  (let bedroom_time := 10
   let kitchen_time := 12
   let living_room_time := 8
   let dining_room_time := 6
   let office_time := 9
   let bathroom_time := 5
   let battery_duration := 8
   3 * bedroom_time + kitchen_time + living_room_time + dining_room_time + office_time + 2 * bathroom_time) / battery_duration = 10 :=
by sorry

end mary_needs_10_charges_to_vacuum_house_l260_260947


namespace total_cookies_sold_l260_260400

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l260_260400


namespace rose_tom_profit_difference_l260_260667

def investment_months (amount: ℕ) (months: ℕ) : ℕ :=
  amount * months

def total_investment_months (john_inv: ℕ) (rose_inv: ℕ) (tom_inv: ℕ) : ℕ :=
  john_inv + rose_inv + tom_inv

def profit_share (investment: ℕ) (total_investment: ℕ) (total_profit: ℕ) : ℤ :=
  (investment * total_profit) / total_investment

theorem rose_tom_profit_difference
  (john_inv rs_per_year: ℕ := 18000 * 12)
  (rose_inv rs_per_9_months: ℕ := 12000 * 9)
  (tom_inv rs_per_8_months: ℕ := 9000 * 8)
  (total_profit: ℕ := 4070):
  profit_share rose_inv (total_investment_months john_inv rose_inv tom_inv) total_profit -
  profit_share tom_inv (total_investment_months john_inv rose_inv tom_inv) total_profit = 370 := 
by
  sorry

end rose_tom_profit_difference_l260_260667


namespace sum_of_digits_of_N_l260_260327

def d (n : ℕ) : ℕ :=
  Nat.divisors n

def f (n : ℕ) : ℝ :=
  (d n).card / (n:ℝ)^(1/3)

noncomputable def N : ℕ :=
  2^3 * 3^2 * 5 * 7

theorem sum_of_digits_of_N : (N.digits.sum = 9) :=
begin
  sorry
end

end sum_of_digits_of_N_l260_260327


namespace polynomial_has_n_real_roots_l260_260275

noncomputable def P_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), (2:ℝ)^k * (Nat.choose (2 * n) (2 * k)) * (x^k) * ((x - 1)^(n - k))

theorem polynomial_has_n_real_roots (n : ℕ) (hn : 0 < n) : ∃ S : Finset ℝ, S.card = n ∧ ∀ x ∈ S, P_n n x = 0 ∧ 0 < x ∧ x < 1 :=
by
  sorry

end polynomial_has_n_real_roots_l260_260275


namespace average_marks_of_passed_l260_260499

theorem average_marks_of_passed
  (total_boys : ℕ)
  (average_all : ℕ)
  (average_failed : ℕ)
  (passed_boys : ℕ)
  (num_boys := 120)
  (avg_all := 37)
  (avg_failed := 15)
  (passed := 110)
  (failed_boys := total_boys - passed_boys)
  (total_marks_all := average_all * total_boys)
  (total_marks_failed := average_failed * failed_boys)
  (total_marks_passed := total_marks_all - total_marks_failed)
  (average_passed := total_marks_passed / passed_boys) :
  average_passed = 39 :=
by
  -- start of proof
  sorry

end average_marks_of_passed_l260_260499


namespace intersection_point_l260_260535

theorem intersection_point (x y : ℚ) 
  (h1 : 3 * y = -2 * x + 6) 
  (h2 : 2 * y = 7 * x - 4) :
  x = 24 / 25 ∧ y = 34 / 25 :=
sorry

end intersection_point_l260_260535


namespace find_positive_X_l260_260040

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l260_260040


namespace div_by_5_mul_diff_l260_260295

theorem div_by_5_mul_diff (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
by
  sorry

end div_by_5_mul_diff_l260_260295


namespace distance_between_foci_of_hyperbola_l260_260533

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : x ^ 2 - 4 * y ^ 2 = 4) :
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt 5 :=
sorry

end distance_between_foci_of_hyperbola_l260_260533


namespace cat_litter_container_weight_l260_260316

theorem cat_litter_container_weight :
  (∀ (cost_container : ℕ) (pounds_per_litterbox : ℕ) (cost_total : ℕ) (days : ℕ),
    cost_container = 21 ∧ pounds_per_litterbox = 15 ∧ cost_total = 210 ∧ days = 210 → 
    ∀ (weeks : ℕ), weeks = days / 7 →
    ∀ (containers : ℕ), containers = cost_total / cost_container →
    ∀ (cost_per_container : ℕ), cost_per_container = cost_total / containers →
    (∃ (pounds_per_container : ℕ), pounds_per_container = cost_container / cost_per_container ∧ pounds_per_container = 3)) :=
by
  intros cost_container pounds_per_litterbox cost_total days
  intros h weeks hw containers hc containers_cost hc_cost
  sorry

end cat_litter_container_weight_l260_260316


namespace evaluate_f_5_minus_f_neg_5_l260_260743

noncomputable def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by 
  sorry

end evaluate_f_5_minus_f_neg_5_l260_260743


namespace convert_scientific_notation_l260_260359

theorem convert_scientific_notation (a : ℝ) (b : ℤ) (h : a = 6.03 ∧ b = 5) : a * 10^b = 603000 := by
  cases h with
  | intro ha hb =>
    rw [ha, hb]
    sorry

end convert_scientific_notation_l260_260359


namespace exists_odd_a_b_and_positive_k_l260_260434

theorem exists_odd_a_b_and_positive_k (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ k > 0 ∧ 2 * m = a^5 + b^5 + k * 2^100 := 
sorry

end exists_odd_a_b_and_positive_k_l260_260434


namespace common_difference_l260_260026

-- Define the arithmetic sequence with general term
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem common_difference (a₁ a₅ a₄ d : ℕ) 
  (h₁ : a₁ + a₅ = 10)
  (h₂ : a₄ = 7)
  (h₅ : a₅ = a₁ + 4 * d)
  (h₄ : a₄ = a₁ + 3 * d) :
  d = 2 :=
by
  sorry

end common_difference_l260_260026


namespace simplify_expression_l260_260480

variable (x : ℝ)

theorem simplify_expression :
  (2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6)) = (8 * x^3 - 4 * x^2 + 6 * x - 24) := 
by 
  sorry

end simplify_expression_l260_260480


namespace knights_count_l260_260321

theorem knights_count (n : ℕ) (h₁ : n = 65) (h₂ : ∀ i, 1 ≤ i → i ≤ n → 
                     (∃ T F, (T = (∑ j in finset.range (i-1), if j < i then 1 else 0) - F)
                              (F = (∑ j in finset.range (i-1), if j >= i then 1 else 0) + 20))) : 
                     (∑ i in finset.filter (λ i, odd i) (finset.filter (λ i, 21 ≤ i ∧ ¬ i > 65) (finset.range 66))) = 23 :=
begin
  sorry
end

end knights_count_l260_260321


namespace correct_statement_l260_260906

theorem correct_statement (a b : ℚ) :
  (|a| = b → a = b) ∧ (|a| > |b| → a > b) ∧ (|a| > b → |a| > |b|) ∧ (|a| = b → a^2 = (-b)^2) ↔ 
  (true ∧ false ∧ false ∧ true) :=
by
  sorry

end correct_statement_l260_260906


namespace time_to_be_100_miles_apart_l260_260249

noncomputable def distance_apart (x : ℝ) : ℝ :=
  Real.sqrt ((12 * x) ^ 2 + (16 * x) ^ 2)

theorem time_to_be_100_miles_apart : ∃ x : ℝ, distance_apart x = 100 ↔ x = 5 :=
by {
  sorry
}

end time_to_be_100_miles_apart_l260_260249


namespace number_of_combinations_l260_260542

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations_l260_260542


namespace general_term_formula_l260_260022

theorem general_term_formula (a S : ℕ → ℝ) (h : ∀ n, S n = (2 / 3) * a n + (1 / 3)) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = -2 * a (n - 1)) →
  ∀ n, a n = (-2)^(n - 1) :=
by
  sorry

end general_term_formula_l260_260022


namespace product_of_three_greater_than_two_or_four_of_others_l260_260730

theorem product_of_three_greater_than_two_or_four_of_others 
  (x : Fin 10 → ℕ) 
  (h_unique : ∀ i j : Fin 10, i ≠ j → x i ≠ x j) 
  (h_positive : ∀ i : Fin 10, 0 < x i) : 
  ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ a b : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ b ≠ i ∧ b ≠ j ∧ b ≠ k → 
      x i * x j * x k > x a * x b) ∨ 
    (∀ a b c d : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ 
      b ≠ i ∧ b ≠ j ∧ b ≠ k ∧ 
      c ≠ i ∧ c ≠ j ∧ c ≠ k ∧ 
      d ≠ i ∧ d ≠ j ∧ d ≠ k → 
      x i * x j * x k > x a * x b * x c * x d) := sorry

end product_of_three_greater_than_two_or_four_of_others_l260_260730


namespace maxValue_of_MF1_MF2_l260_260427

noncomputable def maxProductFociDistances : ℝ :=
  let C : set (ℝ × ℝ) := { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) }
  let F₁ : ℝ × ℝ := (-√(5), 0)
  let F₂ : ℝ × ℝ := (√(5), 0)
  classical.some (maxSetOf (λ (p : ℝ × ℝ), dist p F₁ * dist p F₂) C)

theorem maxValue_of_MF1_MF2 :
  ∃ M : ℝ × ℝ, 
    M ∈ { p | ∃ x y, x^2 / 9 + y^2 / 4 = 1 ∧ p = (x, y) } ∧
    dist M (-√(5), 0) * dist M (√(5), 0) = 9 :=
sorry

end maxValue_of_MF1_MF2_l260_260427


namespace fraction_of_income_from_tips_l260_260824

variable (S T I : ℝ)

theorem fraction_of_income_from_tips (h1 : T = (5 / 2) * S) (h2 : I = S + T) : 
  T / I = 5 / 7 := by
  sorry

end fraction_of_income_from_tips_l260_260824


namespace clara_total_cookies_l260_260405

theorem clara_total_cookies :
  let cookies_per_box1 := 12
  let cookies_per_box2 := 20
  let cookies_per_box3 := 16
  let boxes_sold1 := 50
  let boxes_sold2 := 80
  let boxes_sold3 := 70
  (boxes_sold1 * cookies_per_box1 + boxes_sold2 * cookies_per_box2 + boxes_sold3 * cookies_per_box3) = 3320 :=
by
  sorry

end clara_total_cookies_l260_260405


namespace range_of_a_for_propositions_p_and_q_l260_260406

theorem range_of_a_for_propositions_p_and_q :
  {a : ℝ | ∃ x, (x^2 + 2 * a * x + 4 = 0) ∧ (3 - 2 * a > 1)} = {a | a ≤ -2} := sorry

end range_of_a_for_propositions_p_and_q_l260_260406


namespace remainder_sum_modulo_l260_260272

theorem remainder_sum_modulo :
  (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 :=
by
sorry

end remainder_sum_modulo_l260_260272


namespace frosting_time_difference_l260_260938

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l260_260938


namespace smallest_n_satisfies_conditions_l260_260623

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l260_260623


namespace math_problem_l260_260350

def cond1 (R r a b c p : ℝ) : Prop := R * r = (a * b * c) / (4 * p)
def cond2 (a b c p : ℝ) : Prop := a * b * c ≤ 8 * p^3
def cond3 (a b c p : ℝ) : Prop := p^2 ≤ (3 * (a^2 + b^2 + c^2)) / 4
def cond4 (m_a m_b m_c R : ℝ) : Prop := m_a^2 + m_b^2 + m_c^2 ≤ (27 * R^2) / 4

theorem math_problem (R r a b c p m_a m_b m_c : ℝ) 
  (h1 : cond1 R r a b c p)
  (h2 : cond2 a b c p)
  (h3 : cond3 a b c p)
  (h4 : cond4 m_a m_b m_c R) : 
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ (27 * R^2) / 2 :=
by 
  sorry

end math_problem_l260_260350


namespace find_y_and_y2_l260_260508

theorem find_y_and_y2 (d y y2 : ℤ) (h1 : 3 ^ 2 = 9) (h2 : 3 ^ 4 = 81)
  (h3 : y = 9 + d) (h4 : y2 = 81 + d) (h5 : 81 = 9 + 3 * d) :
  y = 33 ∧ y2 = 105 :=
by
  sorry

end find_y_and_y2_l260_260508


namespace wrong_mark_is_43_l260_260997

theorem wrong_mark_is_43
  (correct_mark : ℕ)
  (wrong_mark : ℕ)
  (num_students : ℕ)
  (avg_increase : ℕ)
  (h_correct : correct_mark = 63)
  (h_num_students : num_students = 40)
  (h_avg_increase : avg_increase = 40 / 2) 
  (h_wrong_avg : (num_students - 1) * (correct_mark + avg_increase) / num_students = (num_students - 1) * (wrong_mark + avg_increase + correct_mark) / num_students) :
  wrong_mark = 43 :=
sorry

end wrong_mark_is_43_l260_260997


namespace rhombus_area_l260_260669

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 11) (h2 : d2 = 16) : (d1 * d2) / 2 = 88 :=
by {
  -- substitution and proof are omitted, proof body would be provided here
  sorry
}

end rhombus_area_l260_260669


namespace price_of_one_rose_l260_260185

theorem price_of_one_rose
  (tulips1 tulips2 tulips3 roses1 roses2 roses3 : ℕ)
  (price_tulip : ℕ)
  (total_earnings : ℕ)
  (R : ℕ) :
  tulips1 = 30 →
  roses1 = 20 →
  tulips2 = 2 * tulips1 →
  roses2 = 2 * roses1 →
  tulips3 = 10 * tulips2 / 100 →  -- simplification of 0.1 * tulips2
  roses3 = 16 →
  price_tulip = 2 →
  total_earnings = 420 →
  (96 * price_tulip + 76 * R) = total_earnings →
  R = 3 :=
by
  intros
  -- Proof will go here
  sorry

end price_of_one_rose_l260_260185


namespace average_stoppage_time_l260_260297

def bus_a_speed_excluding_stoppages := 54 -- kmph
def bus_a_speed_including_stoppages := 45 -- kmph

def bus_b_speed_excluding_stoppages := 60 -- kmph
def bus_b_speed_including_stoppages := 50 -- kmph

def bus_c_speed_excluding_stoppages := 72 -- kmph
def bus_c_speed_including_stoppages := 60 -- kmph

theorem average_stoppage_time :
  (bus_a_speed_excluding_stoppages - bus_a_speed_including_stoppages) / bus_a_speed_excluding_stoppages * 60
  + (bus_b_speed_excluding_stoppages - bus_b_speed_including_stoppages) / bus_b_speed_excluding_stoppages * 60
  + (bus_c_speed_excluding_stoppages - bus_c_speed_including_stoppages) / bus_c_speed_excluding_stoppages * 60
  = 30 / 3 :=
  by sorry

end average_stoppage_time_l260_260297


namespace roots_polynomial_sum_squares_l260_260047

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end roots_polynomial_sum_squares_l260_260047


namespace volleyball_team_math_count_l260_260116

theorem volleyball_team_math_count (total_players taking_physics taking_both : ℕ) 
  (h1 : total_players = 30) 
  (h2 : taking_physics = 15) 
  (h3 : taking_both = 6) 
  (h4 : total_players = 30 ∧ total_players = (taking_physics + (total_players - taking_physics - taking_both))) 
  : (total_players - (taking_physics - taking_both) + taking_both) = 21 := 
by
  sorry

end volleyball_team_math_count_l260_260116


namespace Q_div_P_l260_260204

theorem Q_div_P (P Q : ℤ) (h : ∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 →
  P / (x + 7) + Q / (x * (x - 6)) = (x^2 - x + 15) / (x^3 + x^2 - 42 * x)) :
  Q / P = 7 :=
sorry

end Q_div_P_l260_260204


namespace find_number_l260_260746

theorem find_number (x number : ℝ) (h₁ : 5 - (5 / x) = number + (4 / x)) (h₂ : x = 9) : number = 4 :=
by
  subst h₂
  -- proof steps
  sorry

end find_number_l260_260746


namespace convert_cylindrical_to_rectangular_l260_260850

theorem convert_cylindrical_to_rectangular (r θ z x y : ℝ) (h_r : r = 5) (h_θ : θ = (3 * Real.pi) / 2) (h_z : z = 4)
    (h_x : x = r * Real.cos θ) (h_y : y = r * Real.sin θ) :
    (x, y, z) = (0, -5, 4) :=
by
    sorry

end convert_cylindrical_to_rectangular_l260_260850


namespace max_value_9_l260_260330

noncomputable def max_ab_ac_bc (a b c : ℝ) : ℝ :=
  max (a * b) (max (a * c) (b * c))

theorem max_value_9 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 12) (h_prod : a * b + b * c + c * a = 27) :
  max_ab_ac_bc a b c = 9 :=
sorry

end max_value_9_l260_260330


namespace sarah_flour_total_l260_260195

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def chickpea_flour : ℕ := 3
def whole_wheat_pastry_flour : ℕ := 2

def total_flour : ℕ := rye_flour + whole_wheat_bread_flour + chickpea_flour + whole_wheat_pastry_flour

theorem sarah_flour_total : total_flour = 20 := by
  sorry

end sarah_flour_total_l260_260195


namespace divisible_by_117_l260_260477

theorem divisible_by_117 (n : ℕ) (hn : 0 < n) :
  117 ∣ (3^(2*(n+1)) * 5^(2*n) - 3^(3*n+2) * 2^(2*n)) :=
sorry

end divisible_by_117_l260_260477


namespace ball_bounce_height_l260_260235

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (hₖ : ℕ → ℝ) :
  h₀ = 500 ∧ r = 0.6 ∧ (∀ k, hₖ k = h₀ * r^k) → 
  ∃ k, hₖ k < 3 ∧ k ≥ 22 := 
by
  sorry

end ball_bounce_height_l260_260235


namespace marathon_time_l260_260524

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end marathon_time_l260_260524


namespace product_greater_than_constant_l260_260441

noncomputable def f (x m : ℝ) := Real.log x - (m + 1) * x + (1 / 2) * m * x ^ 2
noncomputable def g (x m : ℝ) := Real.log x - (m + 1) * x

variables {x1 x2 m : ℝ} 
  (h1 : g x1 m = 0)
  (h2 : g x2 m = 0)
  (h3 : x2 > Real.exp 1 * x1)

theorem product_greater_than_constant :
  x1 * x2 > 2 / (Real.exp 1 - 1) :=
sorry

end product_greater_than_constant_l260_260441


namespace not_rain_both_rain_one_exactly_rain_at_least_one_rain_at_most_one_l260_260854

namespace RainProbability

variables (P : Set → ℝ) (A B : Set)
variables (probA : P A = 0.2) (probB : P B = 0.3)
variables (independent : P (A ∩ B) = P A * P B)

open Set

-- 1. Probability that it does not rain in both places A and B
theorem not_rain_both : P (Aᶜ ∩ Bᶜ) = 0.56 := sorry

-- 2. Probability that it rains in exactly one of places A or B
theorem rain_one_exactly : P ((A ∩ Bᶜ) ∪ (Aᶜ ∩ B)) = 0.38 := sorry

-- 3. Probability that it rains in at least one of places A or B
theorem rain_at_least_one : P (A ∪ B) = 0.44 := sorry

-- 4. Probability that it rains in at most one of places A or B
theorem rain_at_most_one : P ((A ∩ Bᶜ) ∪ (Aᶜ ∩ B) ∪ (Aᶜ ∩ Bᶜ)) = 0.94 := sorry

end RainProbability

end not_rain_both_rain_one_exactly_rain_at_least_one_rain_at_most_one_l260_260854


namespace remaining_string_length_l260_260215

theorem remaining_string_length (original_length : ℝ) (given_to_Minyoung : ℝ) (fraction_used : ℝ) :
  original_length = 70 →
  given_to_Minyoung = 27 →
  fraction_used = 7/9 →
  abs (original_length - given_to_Minyoung - fraction_used * (original_length - given_to_Minyoung) - 9.56) < 0.01 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end remaining_string_length_l260_260215


namespace graphs_intersect_once_l260_260849

theorem graphs_intersect_once : 
  ∃! (x : ℝ), |3 * x + 6| = -|4 * x - 3| :=
sorry

end graphs_intersect_once_l260_260849


namespace walter_percent_of_dollar_l260_260816

theorem walter_percent_of_dollar
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (penny_value : Nat := 1)
  (nickel_value : Nat := 5)
  (dime_value : Nat := 10)
  (dollar_value : Nat := 100)
  (total_value := pennies * penny_value + nickels * nickel_value + dimes * dime_value) :
  pennies = 2 ∧ nickels = 3 ∧ dimes = 2 →
  (total_value * 100) / dollar_value = 37 :=
by
  sorry

end walter_percent_of_dollar_l260_260816


namespace gum_sharing_l260_260038

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l260_260038


namespace minimum_value_is_81_l260_260176

noncomputable def minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) : ℝ :=
a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_is_81 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_value a b c h1 h2 h3 h4 = 81 :=
sorry

end minimum_value_is_81_l260_260176


namespace ann_frosting_cakes_l260_260935

theorem ann_frosting_cakes (normalRate sprainedRate cakes : ℕ) (H1 : normalRate = 5) (H2 : sprainedRate = 8) (H3 : cakes = 10) :
  (sprainedRate * cakes) - (normalRate * cakes) = 30 :=
by
  -- Substitute the provided values into the expression
  rw [H1, H2, H3]
  -- Evaluate the expression
  norm_num

end ann_frosting_cakes_l260_260935


namespace area_three_layers_is_nine_l260_260089

-- Define the areas as natural numbers
variable (P Q R S T U V : ℕ)

-- Define the combined area of the rugs
def combined_area_rugs := P + Q + R + 2 * (S + T + U) + 3 * V = 90

-- Define the total area covered by the floor
def total_area_floor := P + Q + R + S + T + U + V = 60

-- Define the area covered by exactly two layers of rug
def area_two_layers := S + T + U = 12

-- Define the area covered by exactly three layers of rug
def area_three_layers := V

-- Prove the area covered by exactly three layers of rug is 9
theorem area_three_layers_is_nine
  (h1 : combined_area_rugs P Q R S T U V)
  (h2 : total_area_floor P Q R S T U V)
  (h3 : area_two_layers S T U) :
  area_three_layers V = 9 := by
  sorry

end area_three_layers_is_nine_l260_260089


namespace acronym_XYZ_length_l260_260683

theorem acronym_XYZ_length :
  let X_length := 2 * Real.sqrt 2
  let Y_length := 1 + 2 * Real.sqrt 2
  let Z_length := 4 + Real.sqrt 5
  X_length + Y_length + Z_length = 5 + 4 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end acronym_XYZ_length_l260_260683


namespace fraction_yellow_surface_area_l260_260828

theorem fraction_yellow_surface_area
  (cube_edge : ℕ)
  (small_cubes : ℕ)
  (yellow_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (fraction_yellow : ℚ) :
  cube_edge = 4 ∧
  small_cubes = 64 ∧
  yellow_cubes = 15 ∧
  total_surface_area = 6 * cube_edge * cube_edge ∧
  yellow_surface_area = 16 ∧
  fraction_yellow = yellow_surface_area / total_surface_area →
  fraction_yellow = 1/6 :=
by
  sorry

end fraction_yellow_surface_area_l260_260828


namespace stream_speed_l260_260617

variables (v_s t_d t_u : ℝ)
variables (D : ℝ) -- Distance is not provided in the problem but assumed for formulation.

theorem stream_speed (h1 : t_u = 2 * t_d) (h2 : v_s = 54 + t_d / t_u) :
  v_s = 18 := 
by
  sorry

end stream_speed_l260_260617


namespace div_coeff_roots_l260_260203

theorem div_coeff_roots :
  ∀ (a b c d e : ℝ), (∀ x, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4)
  → (d / e = -25 / 12) :=
by
  intros a b c d e h
  sorry

end div_coeff_roots_l260_260203


namespace number_writing_number_reading_l260_260242

def ten_million_place := 10^7
def hundred_thousand_place := 10^5
def ten_place := 10

def ten_million := 1 * ten_million_place
def three_hundred_thousand := 3 * hundred_thousand_place
def fifty := 5 * ten_place

def constructed_number := ten_million + three_hundred_thousand + fifty

def read_number := "ten million and thirty thousand and fifty"

theorem number_writing : constructed_number = 10300050 := by
  -- Sketch of proof goes here based on place values
  sorry

theorem number_reading : read_number = "ten million and thirty thousand and fifty" := by
  -- Sketch of proof goes here for the reading method
  sorry

end number_writing_number_reading_l260_260242


namespace smallest_positive_n_l260_260641

noncomputable def smallest_n (n : ℕ) :=
  (∃ k1 : ℕ, 5 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^3) ∧ n > 0

theorem smallest_positive_n :
  ∃ n : ℕ, smallest_n n ∧ ∀ m : ℕ, smallest_n m → n ≤ m := 
sorry

end smallest_positive_n_l260_260641


namespace max_catch_up_distance_l260_260838

/-- 
Given:
  - The total length of the race is 5000 feet.
  - Alex and Max are even for the first 200 feet, so the initial distance between them is 0 feet.
  - On the uphill slope, Alex gets ahead by 300 feet.
  - On the downhill slope, Max gains a lead of 170 feet over Alex, reducing Alex's lead.
  - On the flat section, Alex pulls ahead by 440 feet.

Prove:
  - The distance left for Max to catch up to Alex is 4430 feet.
--/
theorem max_catch_up_distance :
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  total_distance - final_distance = 4430 :=
by
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  have final_distance_calc : final_distance = 570
  sorry
  show total_distance - final_distance = 4430
  sorry

end max_catch_up_distance_l260_260838


namespace find_angle_B_l260_260023

def is_triangle (A B C : ℝ) : Prop :=
A + B > C ∧ B + C > A ∧ C + A > B

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Defining the problem conditions
lemma given_condition : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a := sorry
-- A triangle with sides a, b, c
lemma triangle_property : is_triangle a b c := sorry

-- The equivalent proof problem
theorem find_angle_B (h_triangle : is_triangle a b c) (h_cond : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) : 
    B = π / 6 := sorry

end find_angle_B_l260_260023


namespace volume_difference_l260_260881

-- Define the dimensions of the first bowl
def length1 : ℝ := 14
def width1 : ℝ := 16
def height1 : ℝ := 9

-- Define the dimensions of the second bowl
def length2 : ℝ := 14
def width2 : ℝ := 16
def height2 : ℝ := 4

-- Define the volumes of the two bowls assuming they are rectangular prisms
def volume1 : ℝ := length1 * width1 * height1
def volume2 : ℝ := length2 * width2 * height2

-- Statement to prove the volume difference
theorem volume_difference : volume1 - volume2 = 1120 := by
  sorry

end volume_difference_l260_260881


namespace elysse_bags_per_trip_l260_260130

-- Definitions from the problem conditions
def total_bags : ℕ := 30
def total_trips : ℕ := 5
def bags_per_trip : ℕ := total_bags / total_trips

def carries_same_amount (elysse_bags brother_bags : ℕ) : Prop := elysse_bags = brother_bags

-- Statement to prove
theorem elysse_bags_per_trip :
  ∀ (elysse_bags brother_bags : ℕ), 
  bags_per_trip = elysse_bags + brother_bags → 
  carries_same_amount elysse_bags brother_bags → 
  elysse_bags = 3 := 
by 
  intros elysse_bags brother_bags h1 h2
  sorry

end elysse_bags_per_trip_l260_260130


namespace sin_sum_angle_eq_sqrt15_div5_l260_260142

variable {x : Real}
variable (h1 : 0 < x ∧ x < Real.pi) (h2 : Real.sin (2 * x) = 1 / 5)

theorem sin_sum_angle_eq_sqrt15_div5 : Real.sin (Real.pi / 4 + x) = Real.sqrt 15 / 5 := by
  -- The proof is omitted as instructed.
  sorry

end sin_sum_angle_eq_sqrt15_div5_l260_260142


namespace base_b_digits_l260_260519

theorem base_b_digits (b : ℕ) : b^4 ≤ 500 ∧ 500 < b^5 → b = 4 := by
  intro h
  sorry

end base_b_digits_l260_260519


namespace positive_difference_of_roots_l260_260543

open Polynomial

theorem positive_difference_of_roots (a b c : ℤ) (h_eq : a = 5 ∧ b = -11 ∧ c = -14)
  (h_quad : a ≠ 0) :
  let Δ := b^2 - 4 * a * c,
      root_diff := (Real.sqrt (Δ)) / (a * 2)
  in Δ = 401 ∧ root_diff = Real.sqrt 401 / 5 → 406 = 401 + 5 :=
by
  sorry

end positive_difference_of_roots_l260_260543


namespace price_of_cookie_cookie_price_verification_l260_260944

theorem price_of_cookie 
  (total_spent : ℝ) 
  (cost_per_cupcake : ℝ)
  (num_cupcakes : ℕ)
  (cost_per_doughnut : ℝ)
  (num_doughnuts : ℕ)
  (cost_per_pie_slice : ℝ)
  (num_pie_slices : ℕ)
  (num_cookies : ℕ)
  (total_cookies_cost : ℝ)
  (total_cost : ℝ) :
  (num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  + num_cookies * total_cookies_cost = total_spent) → 
  total_cookies_cost = 0.60 :=
by
  sorry

noncomputable def sophie_cookies_price : ℝ := 
  let total_cost := 33
  let num_cupcakes := 5
  let cost_per_cupcake := 2
  let num_doughnuts := 6
  let cost_per_doughnut := 1
  let num_pie_slices := 4
  let cost_per_pie_slice := 2
  let num_cookies := 15
  let total_spent_on_other_items := 
    num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  let remaining_cost := total_cost - total_spent_on_other_items 
  remaining_cost / num_cookies

theorem cookie_price_verification :
  sophie_cookies_price = 0.60 :=
by
  sorry

end price_of_cookie_cookie_price_verification_l260_260944


namespace isPossible_l260_260682

structure Person where
  firstName : String
  patronymic : String
  surname : String

def conditions (people : List Person) : Prop :=
  people.length = 4 ∧
  ∀ p1 p2 p3 : Person, 
    p1 ∈ people → p2 ∈ people → p3 ∈ people →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    p1.firstName ≠ p2.firstName ∨ p2.firstName ≠ p3.firstName ∨ p1.firstName ≠ p3.firstName ∧
    p1.patronymic ≠ p2.patronymic ∨ p2.patronymic ≠ p3.patronymic ∨ p1.patronymic ≠ p3.patronymic ∧
    p1.surname ≠ p2.surname ∨ p2.surname ≠ p3.surname ∨ p1.surname ≠ p3.surname ∧
  ∀ p1 p2 : Person, 
    p1 ∈ people → p2 ∈ people →
    p1 ≠ p2 →
    p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.surname = p2.surname

theorem isPossible : ∃ people : List Person, conditions people := by
  sorry

end isPossible_l260_260682


namespace geom_cos_sequence_l260_260419

open Real

theorem geom_cos_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) (h : cos (2*b) / cos b = cos (3*b) / cos (2*b)) : b = 180 :=
by
  sorry

end geom_cos_sequence_l260_260419


namespace compute_expression_l260_260120

theorem compute_expression :
  ( (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) )
  /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) )
  = 221 := 
by sorry

end compute_expression_l260_260120


namespace max_n_value_l260_260548

-- Define the arithmetic sequence
variable {a : ℕ → ℤ} (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)

-- Given conditions
variable (h1 : a 1 + a 3 + a 5 = 105)
variable (h2 : a 2 + a 4 + a 6 = 99)

-- Goal: Prove the maximum integer value of n is 10
theorem max_n_value (n : ℕ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) : n ≤ 10 → 
  (∀ m, (0 < m ∧ m ≤ n) → a (2 * m) ≥ 0) → n = 10 := 
sorry

end max_n_value_l260_260548


namespace borrow_years_l260_260528

/-- A person borrows Rs. 5000 at 4% p.a simple interest and lends it at 6% p.a simple interest.
His gain in the transaction per year is Rs. 100. Prove that he borrowed the money for 1 year. --/
theorem borrow_years
  (principal : ℝ)
  (borrow_rate : ℝ)
  (lend_rate : ℝ)
  (gain : ℝ)
  (interest_paid_per_year : ℝ)
  (interest_earned_per_year : ℝ) :
  (principal = 5000) →
  (borrow_rate = 0.04) →
  (lend_rate = 0.06) →
  (gain = 100) →
  (interest_paid_per_year = principal * borrow_rate) →
  (interest_earned_per_year = principal * lend_rate) →
  (interest_earned_per_year - interest_paid_per_year = gain) →
  1 = 1 := 
by
  -- Placeholder for the proof
  sorry

end borrow_years_l260_260528


namespace cycle_selling_price_l260_260665

theorem cycle_selling_price
(C : ℝ := 1900)  -- Cost price of the cycle
(Lp : ℝ := 18)  -- Loss percentage
(S : ℝ := 1558) -- Expected selling price
: (S = C - (Lp / 100) * C) :=
by 
  sorry

end cycle_selling_price_l260_260665


namespace find_w_over_y_l260_260155

theorem find_w_over_y 
  (w x y : ℝ) 
  (h1 : w / x = 2 / 3) 
  (h2 : (x + y) / y = 1.6) : 
  w / y = 0.4 := 
  sorry

end find_w_over_y_l260_260155


namespace max_product_of_distances_l260_260430

-- Definition of an ellipse
def ellipse := {M : ℝ × ℝ // (M.1^2 / 9) + (M.2^2 / 4) = 1}

-- Foci of the ellipse
def F1 : ℝ × ℝ := (-√5, 0)
def F2 : ℝ × ℝ := (√5, 0)

-- Function to calculate distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The main theorem: The maximum value of |MF1| * |MF2| for M on the ellipse is 9
theorem max_product_of_distances (M : ellipse) :
  dist M.val F1 * dist M.val F2 ≤ 9 :=
sorry

end max_product_of_distances_l260_260430


namespace fox_appropriation_l260_260335

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end fox_appropriation_l260_260335


namespace Sydney_initial_rocks_l260_260792

variable (S₀ : ℕ)

def Conner_initial : ℕ := 723
def Sydney_collects_day1 : ℕ := 4
def Conner_collects_day1 : ℕ := 8 * Sydney_collects_day1
def Sydney_collects_day2 : ℕ := 0
def Conner_collects_day2 : ℕ := 123
def Sydney_collects_day3 : ℕ := 2 * Conner_collects_day1
def Conner_collects_day3 : ℕ := 27

def Total_Sydney_collects : ℕ := Sydney_collects_day1 + Sydney_collects_day2 + Sydney_collects_day3
def Total_Conner_collects : ℕ := Conner_collects_day1 + Conner_collects_day2 + Conner_collects_day3

def Total_Sydney_rocks : ℕ := S₀ + Total_Sydney_collects
def Total_Conner_rocks : ℕ := Conner_initial + Total_Conner_collects

theorem Sydney_initial_rocks :
  Total_Conner_rocks = Total_Sydney_rocks → S₀ = 837 :=
by
  sorry

end Sydney_initial_rocks_l260_260792


namespace books_for_sale_l260_260465

theorem books_for_sale (initial_books found_books : ℕ) (h1 : initial_books = 33) (h2 : found_books = 26) :
  initial_books + found_books = 59 :=
by
  sorry

end books_for_sale_l260_260465


namespace graph_not_in_third_quadrant_l260_260797

-- Define the conditions
variable (m : ℝ)
variable (h1 : 0 < m)
variable (h2 : m < 2)

-- Define the graph equation
noncomputable def line_eq (x : ℝ) : ℝ := (m - 2) * x + m

-- The proof problem: the graph does not pass through the third quadrant
theorem graph_not_in_third_quadrant : ¬ ∃ x y : ℝ, (x < 0 ∧ y < 0 ∧ y = (m - 2) * x + m) :=
sorry

end graph_not_in_third_quadrant_l260_260797


namespace surface_area_of_rectangular_solid_l260_260537

-- Conditions
variables {a b c : ℕ}
variables (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
variables (h_volume : a * b * c = 308)

-- Question and Proof Problem
theorem surface_area_of_rectangular_solid :
  2 * (a * b + b * c + c * a) = 226 :=
sorry

end surface_area_of_rectangular_solid_l260_260537


namespace range_of_a_l260_260721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 0 < x₁ → 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) ↔ (1 ≤ a) :=
by
  sorry

end range_of_a_l260_260721


namespace neither_biology_nor_chemistry_l260_260308

def science_club_total : ℕ := 80
def biology_members : ℕ := 50
def chemistry_members : ℕ := 40
def both_members : ℕ := 25

theorem neither_biology_nor_chemistry :
  (science_club_total -
  ((biology_members - both_members) +
  (chemistry_members - both_members) +
  both_members)) = 15 := by
  sorry

end neither_biology_nor_chemistry_l260_260308


namespace women_in_room_l260_260313

theorem women_in_room (M W : ℕ) 
  (h1 : 9 * M = 7 * W) 
  (h2 : M + 5 = 23) : 
  3 * (W - 4) = 57 :=
by
  sorry

end women_in_room_l260_260313


namespace fox_cub_distribution_l260_260333

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l260_260333


namespace range_of_t_minus_1_over_t_minus_3_l260_260152

variable {f : ℝ → ℝ}

-- Function conditions: monotonically decreasing and odd
axiom f_mono_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition on the real number t
variable {t : ℝ}
axiom f_condition : f (t^2 - 2 * t) + f (-3) > 0

-- Question: Prove the range of (t-1)/(t-3)
theorem range_of_t_minus_1_over_t_minus_3 (h : -1 < t ∧ t < 3) : 
  ((t - 1) / (t - 3)) < 1/2 :=
  sorry

end range_of_t_minus_1_over_t_minus_3_l260_260152


namespace cookies_left_after_three_days_l260_260170

theorem cookies_left_after_three_days
  (initial_cookies : ℕ)
  (first_day_fraction_eaten : ℚ)
  (second_day_fraction_eaten : ℚ)
  (initial_value : initial_cookies = 64)
  (first_day_fraction : first_day_fraction_eaten = 3/4)
  (second_day_fraction : second_day_fraction_eaten = 1/2) :
  initial_cookies - (first_day_fraction_eaten * 64) - (second_day_fraction_eaten * ((1 - first_day_fraction_eaten) * 64)) = 8 :=
by
  sorry

end cookies_left_after_three_days_l260_260170


namespace problem_statement_l260_260326

noncomputable def a : ℝ := -0.5
noncomputable def b : ℝ := (1 + Real.sqrt 3) / 2

theorem problem_statement
  (h1 : a^2 = 9 / 36)
  (h2 : b^2 = (1 + Real.sqrt 3)^2 / 8)
  (h3 : a < 0)
  (h4 : b > 0) :
  ∃ (x y z : ℤ), (a - b)^2 = x * Real.sqrt y / z ∧ (x + y + z = 6) :=
sorry

end problem_statement_l260_260326
