import Mathlib

namespace ellipse_intersection_theorem_l3740_374050

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m passing through point (x₀, y₀) -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Theorem about an ellipse with specific properties and its intersection with a line -/
theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.a^2 = 12 ∧ C.b = 2 ∧ (C.a^2 - C.b^2 = 8) ∧ 
  l.m = 1 ∧ l.x₀ = -2 ∧ l.y₀ = 1 →
  (∃ A B : ℝ × ℝ,
    (A.1^2 / 12 + A.2^2 / 4 = 1) ∧
    (B.1^2 / 12 + B.2^2 / 4 = 1) ∧
    (A.2 = A.1 + 3) ∧
    (B.2 = B.1 + 3) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 42 / 2)) :=
by sorry

end ellipse_intersection_theorem_l3740_374050


namespace group_c_marks_is_four_l3740_374042

/-- Represents the examination setup with three groups of questions -/
structure Examination where
  total_questions : ℕ
  group_a_marks : ℕ
  group_b_marks : ℕ
  group_b_questions : ℕ
  group_c_questions : ℕ

/-- Theorem stating that under the given conditions, each question in group C carries 4 marks -/
theorem group_c_marks_is_four (exam : Examination)
  (h_total : exam.total_questions = 100)
  (h_group_a : exam.group_a_marks = 1)
  (h_group_b : exam.group_b_marks = 2)
  (h_group_b_count : exam.group_b_questions = 23)
  (h_group_c_count : exam.group_c_questions = 1)
  (h_group_a_percentage : 
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             4 * exam.group_c_questions)) :
  ∃ (group_c_marks : ℕ), group_c_marks = 4 ∧
    group_c_marks > exam.group_b_marks ∧
    exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) ≥
    (3/5) * (exam.group_a_marks * (exam.total_questions - exam.group_b_questions - exam.group_c_questions) +
             exam.group_b_marks * exam.group_b_questions +
             group_c_marks * exam.group_c_questions) := by
  sorry

end group_c_marks_is_four_l3740_374042


namespace min_value_abs_sum_l3740_374041

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 1 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 1 :=
sorry

end min_value_abs_sum_l3740_374041


namespace max_factors_b_power_n_l3740_374062

def count_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_b_power_n (b n : ℕ+) (h1 : b ≤ 20) (h2 : n = 10) :
  (∃ (b' : ℕ+), b' ≤ 20 ∧ count_factors b' n = 231) ∧
  (∀ (b' : ℕ+), b' ≤ 20 → count_factors b' n ≤ 231) :=
sorry

end max_factors_b_power_n_l3740_374062


namespace female_fraction_is_four_fifths_l3740_374000

/-- Represents a corporation with male and female employees -/
structure Corporation where
  maleEmployees : ℕ
  femaleEmployees : ℕ

/-- The fraction of employees who are at least 35 years old -/
def atLeast35Fraction (c : Corporation) : ℚ :=
  (0.5 * c.maleEmployees + 0.4 * c.femaleEmployees) / (c.maleEmployees + c.femaleEmployees)

/-- The fraction of employees who are females -/
def femaleFraction (c : Corporation) : ℚ :=
  c.femaleEmployees / (c.maleEmployees + c.femaleEmployees)

theorem female_fraction_is_four_fifths (c : Corporation) 
    (h : atLeast35Fraction c = 0.42) : 
    femaleFraction c = 4/5 := by
  sorry

end female_fraction_is_four_fifths_l3740_374000


namespace cost_of_pens_l3740_374081

/-- Given that 150 pens cost $45, prove that 3300 pens cost $990 -/
theorem cost_of_pens (pack_size : ℕ) (pack_cost : ℚ) (desired_amount : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_amount = 3300 →
  (desired_amount : ℚ) * (pack_cost / pack_size) = 990 :=
by sorry

end cost_of_pens_l3740_374081


namespace max_value_theorem_l3740_374063

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 2 :=
by sorry

end max_value_theorem_l3740_374063


namespace total_cost_is_eight_times_shorts_l3740_374072

def football_gear_cost (x : ℝ) : Prop :=
  let shorts := x
  let tshirt := x
  let boots := 4 * x
  let shin_guards := 2 * x
  (shorts + tshirt = 2 * x) ∧
  (shorts + boots = 5 * x) ∧
  (shorts + shin_guards = 3 * x) ∧
  (shorts + tshirt + boots + shin_guards = 8 * x)

theorem total_cost_is_eight_times_shorts :
  ∀ x : ℝ, x > 0 → football_gear_cost x :=
by
  sorry

end total_cost_is_eight_times_shorts_l3740_374072


namespace jerry_fifth_night_earnings_l3740_374009

def jerry_tips (n : ℕ) : List ℝ := [20, 60, 15, 40]
def days_worked : ℕ := 5
def target_average : ℝ := 50

theorem jerry_fifth_night_earnings :
  let total_target : ℝ := days_worked * target_average
  let current_total : ℝ := (jerry_tips days_worked).sum
  let required_fifth_night : ℝ := total_target - current_total
  required_fifth_night = 115 := by sorry

end jerry_fifth_night_earnings_l3740_374009


namespace min_delivery_time_l3740_374004

theorem min_delivery_time (n : Nat) (hn : n = 63) :
  let S := Fin n → Fin n
  (∃ (f : S), Function.Bijective f) →
  (∀ (f : S), Function.Bijective f →
    (∃ (i : Fin n), (i.val + 1) * (f i).val + 1 ≥ 1024)) ∧
  (∃ (f : S), Function.Bijective f ∧
    ∀ (i : Fin n), (i.val + 1) * (f i).val + 1 ≤ 1024) :=
by sorry

end min_delivery_time_l3740_374004


namespace tangent_roots_sine_cosine_ratio_l3740_374046

theorem tangent_roots_sine_cosine_ratio (α β p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (Real.sin (α + β)) / (Real.cos (α - β)) = -p / (q + 1) := by
sorry

end tangent_roots_sine_cosine_ratio_l3740_374046


namespace intersection_A_B_solution_set_quadratic_l3740_374018

-- Define sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := 2*x^2 + 4*x - 6 < 0

-- Theorem for the solution set of the quadratic inequality
theorem solution_set_quadratic : {x | quadratic_inequality x} = B := by sorry

end intersection_A_B_solution_set_quadratic_l3740_374018


namespace triangle_angle_c_value_l3740_374023

/-- Given a triangle ABC with internal angles A, B, and C, and vectors m and n
    satisfying certain conditions, prove that C = 2π/3 -/
theorem triangle_angle_c_value 
  (A B C : ℝ) 
  (triangle_sum : A + B + C = π)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (m_def : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (n_def : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (dot_product : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B)) :
  C = 2 * π / 3 := by
sorry

end triangle_angle_c_value_l3740_374023


namespace fred_dimes_l3740_374043

/-- Proves that if Fred has 90 cents and each dime is worth 10 cents, then Fred has 9 dimes -/
theorem fred_dimes (total_cents : ℕ) (dime_value : ℕ) (h1 : total_cents = 90) (h2 : dime_value = 10) :
  total_cents / dime_value = 9 := by
  sorry

end fred_dimes_l3740_374043


namespace quadratic_roots_properties_l3740_374028

theorem quadratic_roots_properties (a b : ℝ) : 
  (a^2 + 3*a - 2 = 0) → (b^2 + 3*b - 2 = 0) → 
  (a + b = -3) ∧ (a^3 + 3*a^2 + 2*b = -6) := by
  sorry

end quadratic_roots_properties_l3740_374028


namespace triangle_median_inequalities_l3740_374002

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, prove two inequalities involving the medians. -/
theorem triangle_median_inequalities (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 / (b*c) + mb^2 / (c*a) + mc^2 / (a*b) ≥ 9/4) ∧
  ((mb^2 + mc^2 - ma^2) / (b*c) + (mc^2 + ma^2 - mb^2) / (c*a) + (ma^2 + mb^2 - mc^2) / (a*b) ≥ 9/4) :=
by sorry

end triangle_median_inequalities_l3740_374002


namespace sphere_surface_area_l3740_374090

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  (4 * Real.pi * r^2 : ℝ) = 36 * 2^(2/3) * Real.pi ↔ (4/3 * Real.pi * r^3 : ℝ) = V := by
  sorry

#check sphere_surface_area

end sphere_surface_area_l3740_374090


namespace discount_clinic_visits_prove_discount_clinic_visits_l3740_374005

def normal_doctor_charge : ℝ := 200
def discount_percentage : ℝ := 0.7
def savings : ℝ := 80

theorem discount_clinic_visits : ℝ :=
  let discount_clinic_charge := normal_doctor_charge * (1 - discount_percentage)
  let total_paid := normal_doctor_charge - savings
  total_paid / discount_clinic_charge

theorem prove_discount_clinic_visits :
  discount_clinic_visits = 2 := by sorry

end discount_clinic_visits_prove_discount_clinic_visits_l3740_374005


namespace vector_operations_l3740_374099

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Vector addition and scalar multiplication -/
def vec_add (u v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => u i + v i
def scalar_mul (r : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => r * v i

/-- Parallel vectors -/
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), v = scalar_mul k u

theorem vector_operations :
  (vec_add (vec_add (scalar_mul 3 a) b) (scalar_mul (-2) c) = ![0, 6]) ∧
  (∃! (m n : ℝ), a = vec_add (scalar_mul m b) (scalar_mul n c) ∧ m = 5/9 ∧ n = 8/9) ∧
  (∃! (k : ℝ), parallel (vec_add a (scalar_mul k c)) (vec_add (scalar_mul 2 b) (scalar_mul (-1) a)) ∧ k = -16/13) :=
by sorry

end vector_operations_l3740_374099


namespace problem1_problem2_l3740_374019

-- Problem 1
theorem problem1 : Real.sqrt 3 ^ 2 - (2023 + π / 2) ^ 0 - (-1) ^ (-1 : ℤ) = 3 := by sorry

-- Problem 2
theorem problem2 : ¬∃ x : ℝ, 5 * x - 4 > 3 * x ∧ (2 * x - 1) / 3 < x / 2 := by sorry

end problem1_problem2_l3740_374019


namespace fox_coins_l3740_374075

def bridge_crossings (initial_coins : ℕ) : ℕ → ℕ
  | 0 => initial_coins + 10
  | n + 1 => (2 * bridge_crossings initial_coins n) - 50

theorem fox_coins (x : ℕ) : x = 37 → bridge_crossings x 4 = 0 := by
  sorry

end fox_coins_l3740_374075


namespace fixed_point_of_exponential_function_l3740_374036

/-- The function f(x) = a^(x-2015) + 2015 passes through the point (2015, 2016) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2015) + 2015
  f 2015 = 2016 := by
sorry

end fixed_point_of_exponential_function_l3740_374036


namespace calculate_income_l3740_374096

/-- Represents a person's monthly income and expenses -/
structure MonthlyFinances where
  income : ℝ
  household_percent : ℝ
  clothes_percent : ℝ
  medicines_percent : ℝ
  savings : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem calculate_income (finances : MonthlyFinances)
  (household_cond : finances.household_percent = 35)
  (clothes_cond : finances.clothes_percent = 20)
  (medicines_cond : finances.medicines_percent = 5)
  (savings_cond : finances.savings = 15000)
  (total_cond : finances.household_percent + finances.clothes_percent + finances.medicines_percent + (finances.savings / finances.income * 100) = 100) :
  finances.income = 37500 := by
  sorry

end calculate_income_l3740_374096


namespace min_value_of_expression_l3740_374006

theorem min_value_of_expression (x : ℝ) :
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧
  ∀ ε > 0, ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) < 4 + ε :=
by sorry

end min_value_of_expression_l3740_374006


namespace log15_12_equals_fraction_l3740_374034

-- Define the logarithm base 10 (lg) and logarithm base 15
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log15 (x : ℝ) := Real.log x / Real.log 15

-- State the theorem
theorem log15_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log15 12 = (2*a + b) / (1 - a + b) := by sorry

end log15_12_equals_fraction_l3740_374034


namespace sin_cos_2alpha_l3740_374051

def fixed_point : ℝ × ℝ := (4, 2)

def is_on_terminal_side (α : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = fixed_point.1 ∧ r * (Real.sin α) = fixed_point.2

theorem sin_cos_2alpha (α : ℝ) (h : is_on_terminal_side α) : 
  Real.sin (2 * α) + Real.cos (2 * α) = 7/5 := by
sorry

end sin_cos_2alpha_l3740_374051


namespace opposite_of_fraction_l3740_374055

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  -(1 : ℚ) / n = -(1 / n) :=
by sorry

end opposite_of_fraction_l3740_374055


namespace preimage_of_five_one_l3740_374088

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_five_one :
  ∃ (p : ℝ × ℝ), f p = (5, 1) ∧ p = (2, 3) :=
by
  sorry

end preimage_of_five_one_l3740_374088


namespace store_marbles_proof_l3740_374095

/-- The number of marbles initially in the store, given the number of customers,
    marbles bought per customer, and remaining marbles after sales. -/
def initial_marbles (customers : ℕ) (marbles_per_customer : ℕ) (remaining_marbles : ℕ) : ℕ :=
  customers * marbles_per_customer + remaining_marbles

theorem store_marbles_proof :
  initial_marbles 20 15 100 = 400 :=
by sorry

end store_marbles_proof_l3740_374095


namespace prob_green_face_specific_die_l3740_374008

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  total_faces_eq : sides = red_faces + green_faces + blue_faces

/-- The probability of rolling a green face on a colored die -/
def prob_green_face (d : ColoredDie) : ℚ :=
  d.green_faces / d.sides

/-- Theorem: The probability of rolling a green face on a 10-sided die
    with 5 red faces, 3 green faces, and 2 blue faces is 3/10 -/
theorem prob_green_face_specific_die :
  let d : ColoredDie := {
    sides := 10,
    red_faces := 5,
    green_faces := 3,
    blue_faces := 2,
    total_faces_eq := by rfl
  }
  prob_green_face d = 3 / 10 := by
  sorry

end prob_green_face_specific_die_l3740_374008


namespace books_sold_l3740_374053

/-- Given the initial number of books and the remaining number of books,
    prove that the number of books sold is their difference. -/
theorem books_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) 
    (h1 : initial = 115)
    (h2 : remaining = 37)
    (h3 : sold = initial - remaining) : 
  sold = 78 := by
  sorry

end books_sold_l3740_374053


namespace sum_of_numbers_l3740_374003

/-- Given the relationships between Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem sum_of_numbers (mickey jayden coraline : ℕ) 
    (h1 : mickey = jayden + 20)
    (h2 : jayden = coraline - 40)
    (h3 : coraline = 80) : 
  mickey + jayden + coraline = 180 := by
  sorry

end sum_of_numbers_l3740_374003


namespace supermarket_turnover_equation_l3740_374015

/-- Represents the equation for a supermarket's quarterly turnover with monthly growth rate -/
theorem supermarket_turnover_equation (x : ℝ) : 
  200 * (1 + (1 + x) + (1 + x)^2) = 1000 ↔ 
  (2 * (1 + x + (1 + x)^2) = 10 ∧ 
   2 > 0 ∧ 
   10 > 0 ∧ 
   (∀ m : ℕ, m < 3 → (1 + x)^m > 0)) := by
  sorry

end supermarket_turnover_equation_l3740_374015


namespace closet_probability_l3740_374093

/-- The number of shirts in the closet -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the closet -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the closet -/
def num_socks : ℕ := 7

/-- The total number of articles of clothing in the closet -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles to be drawn -/
def draw_count : ℕ := 4

/-- The probability of drawing 2 shirts, 1 pair of shorts, and 1 pair of socks -/
theorem closet_probability : 
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) / 
  Nat.choose total_articles draw_count = 56 / 399 := by
  sorry

end closet_probability_l3740_374093


namespace james_partner_teaching_difference_l3740_374068

/-- Proves that the difference in teaching years between James and his partner is 10 -/
theorem james_partner_teaching_difference :
  ∀ (james_years partner_years : ℕ),
    james_years = 40 →
    james_years + partner_years = 70 →
    partner_years < james_years →
    james_years - partner_years = 10 := by
  sorry

end james_partner_teaching_difference_l3740_374068


namespace average_marks_is_76_l3740_374047

def english_marks : ℕ := 73
def math_marks : ℕ := 69
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 64
def biology_marks : ℕ := 82

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_is_76 : (total_marks : ℚ) / num_subjects = 76 := by
  sorry

end average_marks_is_76_l3740_374047


namespace equation_solution_l3740_374037

theorem equation_solution : ∃ y : ℝ, (16 : ℝ) ^ (2 * y - 4) = (1 / 4 : ℝ) ^ (5 - y) ∧ y = 1 := by
  sorry

end equation_solution_l3740_374037


namespace end_with_one_piece_l3740_374001

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (n : ℕ)
  (pieces : ℕ)

/-- Represents a valid move on the chessboard -/
inductive ValidMove : ChessboardState → ChessboardState → Prop
  | jump {s1 s2 : ChessboardState} :
      s1.n = s2.n ∧ s1.pieces = s2.pieces + 1 → ValidMove s1 s2

/-- Represents a sequence of valid moves -/
def ValidMoveSequence : ChessboardState → ChessboardState → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem stating the condition for ending with one piece -/
theorem end_with_one_piece (n : ℕ) :
  (∃ (final : ChessboardState),
    ValidMoveSequence (ChessboardState.mk n (n^2)) final ∧
    final.pieces = 1) ↔ n % 3 ≠ 0 :=
sorry

end end_with_one_piece_l3740_374001


namespace egg_acceptance_ratio_l3740_374012

/-- Represents the egg processing plant scenario -/
structure EggPlant where
  total_eggs : ℕ  -- Total number of eggs processed per day
  normal_accepted : ℕ  -- Number of eggs normally accepted in a batch
  normal_rejected : ℕ  -- Number of eggs normally rejected in a batch
  additional_accepted : ℕ  -- Additional eggs accepted on the particular day

/-- Defines the conditions of the egg processing plant -/
def egg_plant_conditions (plant : EggPlant) : Prop :=
  plant.total_eggs = 400 ∧
  plant.normal_accepted = 96 ∧
  plant.normal_rejected = 4 ∧
  plant.additional_accepted = 12

/-- Calculates the ratio of accepted to rejected eggs on the particular day -/
def acceptance_ratio (plant : EggPlant) : ℚ :=
  let normal_batches := plant.total_eggs / (plant.normal_accepted + plant.normal_rejected)
  let accepted := normal_batches * plant.normal_accepted + plant.additional_accepted
  let rejected := plant.total_eggs - accepted
  accepted / rejected

/-- Theorem stating that under the given conditions, the acceptance ratio is 99:1 -/
theorem egg_acceptance_ratio (plant : EggPlant) 
  (h : egg_plant_conditions plant) : acceptance_ratio plant = 99 / 1 := by
  sorry


end egg_acceptance_ratio_l3740_374012


namespace expr3_greatest_l3740_374011

def expr1 (x y z : ℝ) := 4 * x^2 - 3 * y + 2 * z
def expr2 (x y z : ℝ) := 6 * x - 2 * y^3 + 3 * z^2
def expr3 (x y z : ℝ) := 2 * x^3 - y^2 * z
def expr4 (x y z : ℝ) := x * y^3 - z^2

theorem expr3_greatest :
  let x : ℝ := 3
  let y : ℝ := 2
  let z : ℝ := 1
  expr3 x y z > expr1 x y z ∧
  expr3 x y z > expr2 x y z ∧
  expr3 x y z > expr4 x y z := by
sorry

end expr3_greatest_l3740_374011


namespace square_circle_union_area_l3740_374083

/-- The area of the union of a square and an inscribed circle -/
theorem square_circle_union_area 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 20) 
  (h2 : circle_radius = 10) 
  (h3 : circle_radius = square_side / 2) : 
  square_side ^ 2 = 400 := by sorry

end square_circle_union_area_l3740_374083


namespace garden_perimeter_l3740_374067

theorem garden_perimeter (garden_width playground_length playground_width : ℝ) : 
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
  2 * (garden_width + (playground_length * playground_width / garden_width)) = 56 :=
by
  sorry

end garden_perimeter_l3740_374067


namespace store_savings_l3740_374064

/-- The difference between the selling price and the store's cost for a pair of pants. -/
def price_difference (selling_price store_cost : ℕ) : ℕ :=
  selling_price - store_cost

/-- Theorem stating that the price difference is 8 dollars given the specific selling price and store cost. -/
theorem store_savings : price_difference 34 26 = 8 := by
  sorry

end store_savings_l3740_374064


namespace officer_3_years_shoe_price_l3740_374082

def full_price : ℝ := 85
def discount_1_year : ℝ := 0.2
def discount_3_years : ℝ := 0.25

def price_after_1_year_discount : ℝ := full_price * (1 - discount_1_year)
def price_after_3_years_discount : ℝ := price_after_1_year_discount * (1 - discount_3_years)

theorem officer_3_years_shoe_price :
  price_after_3_years_discount = 51 :=
sorry

end officer_3_years_shoe_price_l3740_374082


namespace total_squares_5x5_with_2_removed_l3740_374048

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Calculates the total number of squares in a grid --/
def total_squares (g : Grid) : ℕ :=
  sorry

/-- The theorem to prove --/
theorem total_squares_5x5_with_2_removed :
  ∃ (g : Grid), g.size = 5 ∧ g.removed = 2 ∧ total_squares g = 55 :=
sorry

end total_squares_5x5_with_2_removed_l3740_374048


namespace remainder_squared_pred_l3740_374054

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end remainder_squared_pred_l3740_374054


namespace ellipse_triangle_perimeter_l3740_374087

/-- Represents an ellipse with equation x²/16 + y²/9 = 1 -/
structure StandardEllipse where
  a : ℝ := 4
  b : ℝ := 3

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ

/-- Represents a focus of the ellipse -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The perimeter of triangle DEF₂ is 16 -/
theorem ellipse_triangle_perimeter
  (e : StandardEllipse)
  (F₁ F F₂ : Focus)
  (D E : EllipsePoint)
  (h1 : F₁.x < F.x) -- F₁ is the left focus
  (h2 : F₂ = F) -- F₂ is the right focus
  (h3 : D.x^2/16 + D.y^2/9 = 1) -- D is on the ellipse
  (h4 : E.x^2/16 + E.y^2/9 = 1) -- E is on the ellipse
  (h5 : ∃ (t : ℝ), D.x = (1-t)*F₁.x + t*E.x ∧ D.y = (1-t)*F₁.y + t*E.y) -- DE passes through F₁
  : (abs (D.x - F₁.x) + abs (D.y - F₁.y)) + 
    (abs (D.x - F₂.x) + abs (D.y - F₂.y)) +
    (abs (E.x - F₁.x) + abs (E.y - F₁.y)) + 
    (abs (E.x - F₂.x) + abs (E.y - F₂.y)) = 16 := by
  sorry

end ellipse_triangle_perimeter_l3740_374087


namespace system_solution_existence_l3740_374007

theorem system_solution_existence (b : ℝ) :
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*Real.sqrt 2 + 1/4 := by
  sorry

end system_solution_existence_l3740_374007


namespace marble_solution_l3740_374035

/-- Represents the number of marbles each person has -/
structure Marbles where
  selma : ℕ
  merill : ℕ
  elliot : ℕ
  vivian : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.selma = 50 ∧
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.vivian = m.merill + m.elliot + 10

/-- The theorem stating the solution to the marble problem -/
theorem marble_solution (m : Marbles) (h : marble_conditions m) : 
  m.merill = 30 ∧ m.vivian = 55 := by
  sorry

#check marble_solution

end marble_solution_l3740_374035


namespace root_equivalence_l3740_374017

theorem root_equivalence (r : ℝ) : r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end root_equivalence_l3740_374017


namespace symmetry_implies_sum_power_l3740_374039

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposites. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2 ∧ A.1 = -B.1

theorem symmetry_implies_sum_power (m n : ℝ) :
  symmetric_y_axis (m, 3) (4, n) → (m + n)^2023 = -1 := by
  sorry

end symmetry_implies_sum_power_l3740_374039


namespace floor_times_self_72_l3740_374020

theorem floor_times_self_72 :
  ∃ (x : ℝ), x > 0 ∧ (Int.floor x : ℝ) * x = 72 ∧ x = 9 := by
  sorry

end floor_times_self_72_l3740_374020


namespace intersection_sum_l3740_374021

/-- Given two lines that intersect at (2,1), prove that a + b = 5/3 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) →  -- First line equation at (2,1)
  (1 = (1/2) * 2 + b) →  -- Second line equation at (2,1)
  a + b = 5/3 := by
  sorry

end intersection_sum_l3740_374021


namespace first_month_sale_l3740_374058

def sales_data : List ℕ := [6927, 6855, 7230, 6562]
def required_sixth_month_sale : ℕ := 5591
def target_average : ℕ := 6600
def num_months : ℕ := 6

theorem first_month_sale (sales : List ℕ) (sixth_sale target_avg n_months : ℕ)
  (h1 : sales = sales_data)
  (h2 : sixth_sale = required_sixth_month_sale)
  (h3 : target_avg = target_average)
  (h4 : n_months = num_months) :
  ∃ (first_sale : ℕ), 
    (first_sale + sales.sum + sixth_sale) / n_months = target_avg ∧ 
    first_sale = 6435 := by
  sorry

end first_month_sale_l3740_374058


namespace remainder_theorem_polynomial_remainder_l3740_374014

def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 20*x - 8

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 44 :=
sorry

end remainder_theorem_polynomial_remainder_l3740_374014


namespace shaded_area_percentage_l3740_374026

theorem shaded_area_percentage (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 4)
  (h3 : large_side > 0)
  (h4 : small_side > 0)
  (h5 : small_side < large_side) :
  (large_side^2 - small_side^2) / large_side^2 * 100 = 84 :=
by sorry

end shaded_area_percentage_l3740_374026


namespace seating_arrangements_eq_48_l3740_374065

/-- The number of ways to seat 7 people around a round table with constraints -/
def seating_arrangements : ℕ :=
  let total_people : ℕ := 7
  let fixed_people : ℕ := 3  -- Alice, Bob, and Carol
  let remaining_people : ℕ := total_people - fixed_people
  let ways_to_arrange_bob_and_carol : ℕ := 2
  ways_to_arrange_bob_and_carol * (Nat.factorial remaining_people)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 : seating_arrangements = 48 := by
  sorry

end seating_arrangements_eq_48_l3740_374065


namespace line_equation_proof_l3740_374022

theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let line_eq := (y - point.2 = slope * (x - point.1))
  line_eq ↔ (y - 1 = Real.sqrt 3 * (x + 2)) :=
by sorry

end line_equation_proof_l3740_374022


namespace sandy_correct_sums_l3740_374094

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
  c + i = 30 →
  3 * c - 2 * i = 55 →
  c = 23 :=
by sorry

end sandy_correct_sums_l3740_374094


namespace intersection_M_N_l3740_374044

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_M_N_l3740_374044


namespace sum_of_even_factors_420_l3740_374076

def sumOfEvenFactors (n : ℕ) : ℕ := sorry

theorem sum_of_even_factors_420 :
  sumOfEvenFactors 420 = 1152 := by sorry

end sum_of_even_factors_420_l3740_374076


namespace problem_solution_l3740_374052

theorem problem_solution (x y : ℝ) : 
  x > 0 → 
  y > 0 → 
  x / 100 * y = 5 → 
  y = 2 * x + 10 → 
  x = 14 := by
sorry

end problem_solution_l3740_374052


namespace greatest_integer_for_integer_fraction_l3740_374079

theorem greatest_integer_for_integer_fraction : 
  (∀ y : ℤ, y > 35 → ¬(∃ n : ℤ, (y^2 + 2*y + 7) / (y - 4) = n)) ∧ 
  (∃ n : ℤ, (35^2 + 2*35 + 7) / (35 - 4) = n) := by
  sorry

end greatest_integer_for_integer_fraction_l3740_374079


namespace count_valid_quadruples_l3740_374045

def valid_quadruple (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
  a^2 + b^2 + c^2 + d^2 = 9 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81

theorem count_valid_quadruples :
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ q ∈ s, valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2) ∧
    (∀ a b c d, valid_quadruple a b c d → (a, b, c, d) ∈ s) ∧
    s.card = 15 :=
sorry

end count_valid_quadruples_l3740_374045


namespace polar_to_rectangular_conversion_l3740_374059

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end polar_to_rectangular_conversion_l3740_374059


namespace line_points_k_value_l3740_374030

/-- Given a line containing the points (0, 4), (7, k), and (21, -2), prove that k = 2 -/
theorem line_points_k_value (k : ℝ) : 
  (∀ t : ℝ, ∃ x y : ℝ, x = t * 7 ∧ y = t * (k - 4) + 4) → 
  (∃ t : ℝ, 21 = t * 7 ∧ -2 = t * (k - 4) + 4) → 
  k = 2 := by
  sorry

end line_points_k_value_l3740_374030


namespace evelyns_bottle_caps_l3740_374080

/-- The problem of Evelyn's bottle caps -/
theorem evelyns_bottle_caps
  (initial : ℕ)            -- Initial number of bottle caps
  (found : ℕ)              -- Number of bottle caps found
  (total : ℕ)              -- Total number of bottle caps at the end
  (h1 : found = 63)        -- Evelyn found 63 bottle caps
  (h2 : total = 81)        -- Evelyn ended up with 81 bottle caps in total
  (h3 : total = initial + found) -- The total is the sum of initial and found bottle caps
  : initial = 18 :=
by sorry

end evelyns_bottle_caps_l3740_374080


namespace perimeter_after_adding_tiles_l3740_374010

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter + 4 }

/-- The theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial_config : TileConfiguration)
  (h1 : initial_config.tiles = 8)
  (h2 : initial_config.perimeter = 14) :
  ∃ (final_config : TileConfiguration),
    final_config = add_tiles initial_config 2 ∧
    final_config.perimeter = 18 := by
  sorry


end perimeter_after_adding_tiles_l3740_374010


namespace polynomial_division_quotient_l3740_374061

theorem polynomial_division_quotient : 
  let dividend : Polynomial ℚ := 8 * X^4 - 4 * X^3 + 3 * X^2 - 5 * X - 10
  let divisor : Polynomial ℚ := X^2 + 3 * X + 2
  let quotient : Polynomial ℚ := 8 * X^2 - 28 * X + 89
  dividend = divisor * quotient + (dividend.mod divisor) := by
  sorry

end polynomial_division_quotient_l3740_374061


namespace fraction_value_l3740_374016

theorem fraction_value (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end fraction_value_l3740_374016


namespace marathon_remainder_yards_l3740_374029

/-- Proves that the remainder yards after 15 marathons is 1500 --/
theorem marathon_remainder_yards :
  let marathons : ℕ := 15
  let miles_per_marathon : ℕ := 26
  let yards_per_marathon : ℕ := 385
  let yards_per_mile : ℕ := 1760
  let total_yards := marathons * (miles_per_marathon * yards_per_mile + yards_per_marathon)
  let full_miles := total_yards / yards_per_mile
  let remainder_yards := total_yards % yards_per_mile
  remainder_yards = 1500 := by sorry

end marathon_remainder_yards_l3740_374029


namespace last_digit_of_3_power_10_l3740_374078

theorem last_digit_of_3_power_10 : ∃ n : ℕ, 3^10 ≡ 9 [ZMOD 10] :=
sorry

end last_digit_of_3_power_10_l3740_374078


namespace least_possible_difference_l3740_374025

theorem least_possible_difference (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 3 → 
  ∀ w, (∃ a b c : ℤ, Even a ∧ Odd b ∧ Odd c ∧ a < b ∧ b < c ∧ b - a > 3 ∧ c - a = w) → w ≥ 7 :=
by sorry

end least_possible_difference_l3740_374025


namespace quadratic_function_minimum_l3740_374024

/-- Given a quadratic function f(x) = ax^2 + bx where a > 0 and b > 0,
    if the slope of the tangent line at x = 1 is 2,
    then the minimum value of (8a + b) / (ab) is 9. -/
theorem quadratic_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a + b = 2) → (∀ x y : ℝ, x > 0 ∧ y > 0 → (8 * x + y) / (x * y) ≥ 9) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (8 * x + y) / (x * y) = 9) := by
  sorry

#check quadratic_function_minimum

end quadratic_function_minimum_l3740_374024


namespace fraction_equality_l3740_374027

theorem fraction_equality : (20 * 2 + 10) / (5 + 3 - 1) = 50 / 7 := by
  sorry

end fraction_equality_l3740_374027


namespace symmetry_implies_a_eq_neg_one_l3740_374073

/-- A function f is symmetric about the line x = c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- The main theorem -/
theorem symmetry_implies_a_eq_neg_one :
  let f := fun (x : ℝ) => Real.sin (2 * x) + a * Real.cos (2 * x)
  SymmetricAbout f (-π/8) → a = -1 := by
  sorry

end symmetry_implies_a_eq_neg_one_l3740_374073


namespace decrypt_ciphertext_l3740_374060

-- Define the encryption function
def encrypt (x : ℕ) : ℕ := 2^x - 2

-- State the theorem
theorem decrypt_ciphertext (y : ℕ) : 
  y = 1022 → ∃ x : ℕ, encrypt x = y ∧ x = 10 := by
  sorry

end decrypt_ciphertext_l3740_374060


namespace circle_equation_l3740_374070

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop :=
  x + 2*y = 0

theorem circle_equation :
  ∀ a : ℝ,
  a < 0 →
  (∃ x y : ℝ, circle_C a x y ∧ tangent_line x y) →
  (∀ x y : ℝ, circle_C a x y ↔ (x + 5)^2 + y^2 = 5) :=
sorry

end circle_equation_l3740_374070


namespace martin_correct_is_40_l3740_374085

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of additional questions Kelsey answered correctly compared to Campbell -/
def kelsey_additional : ℕ := 8

/-- The number of fewer questions Martin answered correctly compared to Kelsey -/
def martin_fewer : ℕ := 3

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := campbell_correct + kelsey_additional - martin_fewer

theorem martin_correct_is_40 : martin_correct = 40 := by
  sorry

end martin_correct_is_40_l3740_374085


namespace log_monotonic_l3740_374086

-- Define the logarithmic function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_monotonic : 
  ∀ x y : ℝ, x > 0 → y > 0 → x < y → log x < log y :=
by
  sorry

end log_monotonic_l3740_374086


namespace line_intersects_circle_l3740_374091

/-- The line l with equation y = kx - 3k intersects the circle C with equation x^2 + y^2 - 4x = 0 for any real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  y = k * x - 3 * k ∧ x^2 + y^2 - 4 * x = 0 := by
  sorry

end line_intersects_circle_l3740_374091


namespace subtraction_of_decimals_l3740_374071

theorem subtraction_of_decimals :
  25.52 - 3.248 - 1.004 = 21.268 := by
  sorry

end subtraction_of_decimals_l3740_374071


namespace largest_prime_factor_of_expression_l3740_374056

def expression : ℕ := 18^3 + 15^4 - 3^7

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 19 := by
  sorry

end largest_prime_factor_of_expression_l3740_374056


namespace weight_of_b_l3740_374074

theorem weight_of_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 46) :
  B = 37 := by sorry

end weight_of_b_l3740_374074


namespace anita_gave_five_apples_l3740_374097

/-- Represents the number of apples Tessa initially had -/
def initial_apples : ℕ := 4

/-- Represents the number of apples Tessa now has -/
def current_apples : ℕ := 9

/-- Represents the number of apples Anita gave Tessa -/
def apples_from_anita : ℕ := current_apples - initial_apples

theorem anita_gave_five_apples : apples_from_anita = 5 := by
  sorry

end anita_gave_five_apples_l3740_374097


namespace valid_sequences_12_l3740_374049

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def valid_sequences (n : ℕ) : ℕ :=
  fibonacci (n + 2)

theorem valid_sequences_12 :
  valid_sequences 12 = 377 :=
by sorry

#eval valid_sequences 12

end valid_sequences_12_l3740_374049


namespace points_two_units_from_negative_three_l3740_374038

theorem points_two_units_from_negative_three :
  ∀ x : ℝ, |(-3) - x| = 2 ↔ x = -5 ∨ x = -1 := by
sorry

end points_two_units_from_negative_three_l3740_374038


namespace sum_of_repeating_decimals_l3740_374077

-- Define repeating decimals
def repeating_decimal_3 : ℚ := 1 / 3
def repeating_decimal_27 : ℚ := 3 / 11

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_3 + repeating_decimal_27 = 20 / 33 := by
  sorry

end sum_of_repeating_decimals_l3740_374077


namespace f_properties_l3740_374031

noncomputable def f (x : ℝ) := x^2 / Real.log x

theorem f_properties :
  let e := Real.exp 1
  ∀ x ∈ Set.Icc (Real.exp (1/4)) e,
    (∀ y ∈ Set.Icc (Real.exp (1/4)) e, f y ≤ f e) ∧
    (f (Real.sqrt e) ≤ f x) ∧
    (∃ t ∈ Set.Icc (2/(e^2)) (1/e), 
      (∃ x₁ ∈ Set.Icc (1/e) 1, t * f x₁ = x₁) ∧
      (∃ x₂ ∈ Set.Ioc 1 (e^2), t * f x₂ = x₂) ∧
      (∀ s < 2/(e^2), ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂) ∧
      (∀ s ≥ 1/e, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  x₂ ∈ Set.Icc (1/e) 1 ∪ Set.Ioc 1 (e^2) ∧
                                  s * f x₁ = x₁ ∧ s * f x₂ = x₂)) :=
by sorry

end f_properties_l3740_374031


namespace job_completion_time_l3740_374032

/-- Given that:
    - A can do a job in 45 days
    - A and B working together can finish 4 times the amount of work in 72 days
    Prove that B can do the job alone in 30 days -/
theorem job_completion_time (a_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) (b_time : ℝ) :
  a_time = 45 →
  combined_time = 72 →
  combined_work = 4 →
  (1 / a_time + 1 / b_time) * combined_time = combined_work →
  b_time = 30 := by
  sorry

end job_completion_time_l3740_374032


namespace max_product_of_functions_l3740_374084

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_of_functions (h k : ℝ → ℝ) 
  (h_range : ∀ x, h x ∈ Set.Icc (-3) 5) 
  (k_range : ∀ x, k x ∈ Set.Icc (-1) 4) : 
  (∃ x y, h x * k y = 20) ∧ (∀ x y, h x * k y ≤ 20) := by
  sorry

end max_product_of_functions_l3740_374084


namespace range_of_a_for_false_quadratic_inequality_l3740_374057

theorem range_of_a_for_false_quadratic_inequality :
  (∃ a : ℝ, ∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ 
  (∃ a : ℝ, -2 < a ∧ a < 2) :=
by sorry

end range_of_a_for_false_quadratic_inequality_l3740_374057


namespace project_time_ratio_l3740_374040

theorem project_time_ratio (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  2 * kate_hours + kate_hours + (kate_hours + 85) = total_hours →
  (2 * kate_hours) / (kate_hours + 85) = 1 / 3 :=
by sorry

end project_time_ratio_l3740_374040


namespace adams_father_deposit_l3740_374013

/-- Calculates the total amount after a given number of years, given an initial deposit,
    annual interest rate, and immediate withdrawal of interest. -/
def totalAmount (initialDeposit : ℝ) (interestRate : ℝ) (years : ℝ) : ℝ :=
  initialDeposit + (initialDeposit * interestRate * years)

/-- Proves that given an initial deposit of $2000 with an 8% annual interest rate,
    where interest is withdrawn immediately upon receipt, the total amount after 2.5 years
    will be $2400. -/
theorem adams_father_deposit : totalAmount 2000 0.08 2.5 = 2400 := by
  sorry

end adams_father_deposit_l3740_374013


namespace david_found_correct_l3740_374069

/-- The amount of money David found on the street -/
def david_found : ℕ := 12

/-- The initial amount of money Evan had -/
def evan_initial : ℕ := 1

/-- The cost of the watch -/
def watch_cost : ℕ := 20

/-- The amount Evan still needs after receiving money from David -/
def evan_still_needs : ℕ := 7

/-- Theorem stating that the amount David found is correct -/
theorem david_found_correct : 
  david_found = watch_cost - evan_still_needs - evan_initial :=
by sorry

end david_found_correct_l3740_374069


namespace gcd_and_polynomial_evaluation_l3740_374066

theorem gcd_and_polynomial_evaluation :
  (Nat.gcd 72 168 = 24) ∧
  (Nat.gcd 98 280 = 14) ∧
  (let f : ℤ → ℤ := fun x => x^5 + x^3 + x^2 + x + 1;
   f 3 = 283) := by
  sorry

end gcd_and_polynomial_evaluation_l3740_374066


namespace rational_coefficient_terms_count_l3740_374089

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (fun (x y : ℝ) => x * Real.rpow 3 (1/4) + y * Real.rpow 5 (1/3)) ^ 400
  let total_terms := 401
  let rational_coeff_count := Finset.filter (fun k => 
    (k % 4 = 0) ∧ ((400 - k) % 3 = 0)
  ) (Finset.range (total_terms))
  34

#check rational_coefficient_terms_count

end rational_coefficient_terms_count_l3740_374089


namespace selling_price_calculation_l3740_374098

theorem selling_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 22500 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (1 - discount_rate) * (cost_price * (1 + profit_rate)) = 24300 →
  cost_price * (1 + profit_rate) / (1 - discount_rate) = 27000 := by
  sorry

#check selling_price_calculation

end selling_price_calculation_l3740_374098


namespace max_value_product_l3740_374092

theorem max_value_product (x y z : ℝ) 
  (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z) 
  (sum_eq_three : x + y + z = 3) : 
  (x^3 - x*y^2 + y^3) * (x^3 - x*z^2 + z^3) * (y^3 - y*z^2 + z^3) ≤ 2916/2187 := by
  sorry

end max_value_product_l3740_374092


namespace quadratic_inequality_solution_set_l3740_374033

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | a * x^2 - 2 * (a - 1) * x + a ≤ 0}
  (a > 1/2 → solution_set = ∅) ∧
  (a = 1/2 → solution_set = {-1}) ∧
  (0 < a ∧ a < 1/2 → solution_set = Set.Icc ((a - 1 - Real.sqrt (1 - 2*a)) / a) ((a - 1 + Real.sqrt (1 - 2*a)) / a)) ∧
  (a = 0 → solution_set = Set.Iic 0) ∧
  (a < 0 → solution_set = Set.Iic ((a - 1 + Real.sqrt (1 - 2*a)) / a) ∪ Set.Ici ((a - 1 - Real.sqrt (1 - 2*a)) / a)) :=
by sorry

end quadratic_inequality_solution_set_l3740_374033
