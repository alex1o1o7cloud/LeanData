import Mathlib

namespace quadratic_equation_sum_l754_75430

theorem quadratic_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 25 = 0 ↔ (x + a)^2 = b) → a + b = -5 := by
  sorry

end quadratic_equation_sum_l754_75430


namespace three_workers_completion_time_l754_75496

/-- The time taken for three workers to complete a task together, given their individual completion times -/
theorem three_workers_completion_time 
  (x_time y_time z_time : ℝ) 
  (hx : x_time = 30) 
  (hy : y_time = 45) 
  (hz : z_time = 60) : 
  (1 / x_time + 1 / y_time + 1 / z_time)⁻¹ = 180 / 13 := by
  sorry

#check three_workers_completion_time

end three_workers_completion_time_l754_75496


namespace partial_fraction_decomposition_l754_75456

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → 
      (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) ∧
    C = 32/9 ∧ D = 13/9 := by
  sorry

end partial_fraction_decomposition_l754_75456


namespace f_derivative_at_one_l754_75453

noncomputable def f' (f'1 : ℝ) : ℝ → ℝ := fun x ↦ 2 * f'1 / x - 1

theorem f_derivative_at_one :
  ∃ f'1 : ℝ, (f' f'1) 1 = 1 :=
sorry

end f_derivative_at_one_l754_75453


namespace marco_juice_mixture_l754_75412

/-- Calculates the remaining mixture after giving some away -/
def remaining_mixture (apple_juice orange_juice given_away : ℚ) : ℚ :=
  apple_juice + orange_juice - given_away

/-- Proves that the remaining mixture is 13/4 gallons -/
theorem marco_juice_mixture :
  let apple_juice : ℚ := 4
  let orange_juice : ℚ := 7/4
  let given_away : ℚ := 5/2
  remaining_mixture apple_juice orange_juice given_away = 13/4 := by
sorry

end marco_juice_mixture_l754_75412


namespace algebraic_expression_value_l754_75442

theorem algebraic_expression_value (a b : ℝ) :
  2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18 →
  9 * b - 6 * a + 2 = 32 := by
sorry

end algebraic_expression_value_l754_75442


namespace unique_three_digit_factorial_sum_l754_75402

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digits_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  digits_factorial_sum n = n ∧
  (n / 100 = 3 ∨ (n / 10) % 10 = 3 ∨ n % 10 = 3) ∧
  n = 145 :=
sorry

end unique_three_digit_factorial_sum_l754_75402


namespace secret_spreading_l754_75415

/-- 
Theorem: Secret Spreading
Given:
- On day 0 (Monday), one person knows a secret.
- Each day, every person who knows the secret tells two new people.
- The number of people who know the secret on day n is 2^(n+1) - 1.

Prove: It takes 9 days for 1023 people to know the secret.
-/
theorem secret_spreading (n : ℕ) : 
  (2^(n+1) - 1 = 1023) → n = 9 := by
  sorry

#check secret_spreading

end secret_spreading_l754_75415


namespace rs_length_l754_75406

/-- Right-angled triangle PQR with perpendiculars to PQ at P and QR at R meeting at S -/
structure SpecialTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  pq_length : dist P Q = 6
  qr_length : dist Q R = 8
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  perp_at_p : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) = 0
  perp_at_r : (S.1 - R.1) * (Q.1 - R.1) + (S.2 - R.2) * (Q.2 - R.2) = 0

/-- The length of RS in the special triangle is 8 -/
theorem rs_length (t : SpecialTriangle) : dist t.R t.S = 8 := by
  sorry

end rs_length_l754_75406


namespace circle_equation_l754_75468

/-- Given a real number a, prove that the equation a²x² + (a+2)y² + 4x + 8y + 5a = 0
    represents a circle with center (-2, -4) and radius 5 if and only if a = -1 -/
theorem circle_equation (a : ℝ) :
  (∃ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0) ∧
  (∀ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0 ↔
    (x + 2)^2 + (y + 4)^2 = 25) ↔
  a = -1 :=
sorry

end circle_equation_l754_75468


namespace inequality_proof_l754_75405

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) := by
  sorry

end inequality_proof_l754_75405


namespace professors_seating_arrangements_l754_75474

/-- Represents the seating arrangement problem with professors and students. -/
structure SeatingArrangement where
  totalChairs : Nat
  numStudents : Nat
  numProfessors : Nat
  professorsBetweenStudents : Bool

/-- Calculates the number of ways professors can choose their chairs. -/
def waysToChooseChairs (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating that the number of ways to choose chairs is 24 for the given problem. -/
theorem professors_seating_arrangements
  (arrangement : SeatingArrangement)
  (h1 : arrangement.totalChairs = 11)
  (h2 : arrangement.numStudents = 7)
  (h3 : arrangement.numProfessors = 4)
  (h4 : arrangement.professorsBetweenStudents = true) :
  waysToChooseChairs arrangement = 24 := by
  sorry

end professors_seating_arrangements_l754_75474


namespace sum_of_fractions_equals_one_l754_75449

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 0) :
  (a^3 * b^3) / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  (a^3 * c^3) / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  (b^3 * c^3) / ((b^3 - a^2 * c) * (c^3 - a^2 * b)) = 1 := by
  sorry

end sum_of_fractions_equals_one_l754_75449


namespace smallest_binary_multiple_of_225_l754_75427

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_225 :
  (∀ m : ℕ, m < 111111100 → ¬(225 ∣ m ∧ is_binary_number m)) ∧
  (225 ∣ 111111100 ∧ is_binary_number 111111100) :=
sorry

end smallest_binary_multiple_of_225_l754_75427


namespace apple_sale_total_l754_75457

theorem apple_sale_total (red_apples : ℕ) (ratio_red : ℕ) (ratio_green : ℕ) : 
  red_apples = 32 → 
  ratio_red = 8 → 
  ratio_green = 3 → 
  red_apples + (red_apples * ratio_green / ratio_red) = 44 := by
sorry

end apple_sale_total_l754_75457


namespace magnitude_of_z_l754_75439

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs z = 2 := by
  sorry

end magnitude_of_z_l754_75439


namespace prob_one_six_max_l754_75410

/-- The probability of rolling exactly one six when rolling n dice -/
def prob_one_six (n : ℕ) : ℚ :=
  (n : ℚ) * (5 ^ (n - 1) : ℚ) / (6 ^ n : ℚ)

/-- The statement that the probability of rolling exactly one six is maximized for 5 or 6 dice -/
theorem prob_one_six_max :
  (∀ k : ℕ, prob_one_six k ≤ prob_one_six 5) ∧
  (prob_one_six 5 = prob_one_six 6) ∧
  (∀ k : ℕ, k > 6 → prob_one_six k < prob_one_six 6) :=
sorry

end prob_one_six_max_l754_75410


namespace inequality_proof_l754_75495

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2*b + 2/(a + 1)) * (b + 2*a + 2/(b + 1)) ≥ 16 := by
  sorry

end inequality_proof_l754_75495


namespace four_digit_numbers_proof_l754_75438

theorem four_digit_numbers_proof (A B : ℕ) : 
  (1000 ≤ A) ∧ (A < 10000) ∧ 
  (1000 ≤ B) ∧ (B < 10000) ∧ 
  (Real.log A / Real.log 10 = 3 + Real.log 4 / Real.log 10) ∧
  (B.div 1000 + B % 10 = 10) ∧
  (B = A / 2 - 21) →
  (A = 4000 ∧ B = 1979) := by
sorry

end four_digit_numbers_proof_l754_75438


namespace value_of_x_l754_75418

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 4)
  (h2 : y = z / 3)
  (h3 : z = 90) :
  x = 7.5 := by
sorry

end value_of_x_l754_75418


namespace midpoint_sum_l754_75467

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 7) and B = (x, y), prove that x + y = 5. -/
theorem midpoint_sum (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (7 + y) / 2 → 
  x + y = 5 := by
sorry

end midpoint_sum_l754_75467


namespace consecutive_product_divisibility_l754_75458

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 8 * m) →
  ¬ (∀ m : ℤ, n = 64 * m) :=
by sorry

end consecutive_product_divisibility_l754_75458


namespace binary_multiplication_theorem_l754_75434

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : ℕ :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true]  -- 101101₂
  let b := [true, true, false, true]               -- 1101₂
  let product := [true, false, false, false, true, false, false, false, true, true, true]  -- 10001000111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

end binary_multiplication_theorem_l754_75434


namespace perpendicular_planes_from_lines_l754_75464

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_lines_l754_75464


namespace product_xyz_equals_negative_one_l754_75407

theorem product_xyz_equals_negative_one (x y z : ℝ) 
  (h1 : x + 1/y = 3) (h2 : y + 1/z = 3) : x * y * z = -1 := by
  sorry

end product_xyz_equals_negative_one_l754_75407


namespace license_plate_difference_l754_75471

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible license plates in Alphazia -/
def alphazia_plates : ℕ := num_letters^4 * num_digits^3

/-- The number of possible license plates in Betaland -/
def betaland_plates : ℕ := num_letters^5 * num_digits^2

/-- The difference in the number of possible license plates between Alphazia and Betaland -/
def plate_difference : ℤ := alphazia_plates - betaland_plates

theorem license_plate_difference :
  plate_difference = -731161600 := by sorry

end license_plate_difference_l754_75471


namespace scout_sale_profit_l754_75447

/-- Represents the scout troop's candy bar sale scenario -/
structure CandyBarSale where
  total_bars : ℕ
  purchase_price : ℚ
  sold_bars : ℕ
  selling_price : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  sale.selling_price * sale.sold_bars - sale.purchase_price * sale.total_bars

/-- The specific candy bar sale scenario from the problem -/
def scout_sale : CandyBarSale :=
  { total_bars := 2000
  , purchase_price := 3 / 4
  , sold_bars := 1950
  , selling_price := 2 / 3 }

/-- Theorem stating that the profit for the scout troop's candy bar sale is -200 -/
theorem scout_sale_profit :
  calculate_profit scout_sale = -200 := by
  sorry


end scout_sale_profit_l754_75447


namespace arithmetic_to_geometric_ratio_l754_75483

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem arithmetic_to_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d ∧ 
  d ≠ 0 ∧
  (∀ n : ℕ, a n ≠ 0) ∧
  ((is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 1))) ∨
   (is_geometric_sequence (λ n => a n) ∧ is_geometric_sequence (λ n => a (n + 2))) ∨
   (is_geometric_sequence (λ n => a (n + 1)) ∧ is_geometric_sequence (λ n => a (n + 2)))) →
  a 0 / d = 1 ∨ a 0 / d = -4 := by
sorry

end arithmetic_to_geometric_ratio_l754_75483


namespace pony_lesson_cost_l754_75487

/-- The cost per lesson for Andrea's pony, given the following conditions:
  * Monthly pasture rent is $500
  * Daily food cost is $10
  * There are two lessons per week
  * Total annual expenditure on the pony is $15890
-/
theorem pony_lesson_cost : 
  let monthly_pasture_rent : ℕ := 500
  let daily_food_cost : ℕ := 10
  let lessons_per_week : ℕ := 2
  let total_annual_cost : ℕ := 15890
  let annual_pasture_cost : ℕ := monthly_pasture_rent * 12
  let annual_food_cost : ℕ := daily_food_cost * 365
  let annual_lessons : ℕ := lessons_per_week * 52
  let lesson_cost : ℕ := (total_annual_cost - (annual_pasture_cost + annual_food_cost)) / annual_lessons
  lesson_cost = 60 :=
by sorry

end pony_lesson_cost_l754_75487


namespace vegetables_sold_mass_l754_75452

/-- Proves that given 15 kg of carrots, 13 kg of zucchini, and 8 kg of broccoli,
    if a merchant sells half of the total vegetables, the mass of vegetables sold is 18 kg. -/
theorem vegetables_sold_mass 
  (carrots : ℝ) 
  (zucchini : ℝ) 
  (broccoli : ℝ) 
  (h1 : carrots = 15)
  (h2 : zucchini = 13)
  (h3 : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

#check vegetables_sold_mass

end vegetables_sold_mass_l754_75452


namespace final_employee_count_l754_75417

/-- Represents the workforce of Company X throughout the year --/
structure CompanyWorkforce where
  initial_total : ℕ
  initial_female : ℕ
  second_quarter_total : ℕ
  second_quarter_female : ℕ
  third_quarter_total : ℕ
  third_quarter_female : ℕ
  final_total : ℕ
  final_female : ℕ

/-- Theorem stating the final number of employees given the workforce changes --/
theorem final_employee_count (w : CompanyWorkforce) : w.final_total = 700 :=
  by
  have h1 : w.initial_female = (60 : ℚ) / 100 * w.initial_total := by sorry
  have h2 : w.second_quarter_total = w.initial_total + 30 := by sorry
  have h3 : w.second_quarter_female = w.initial_female := by sorry
  have h4 : w.second_quarter_female = (57 : ℚ) / 100 * w.second_quarter_total := by sorry
  have h5 : w.third_quarter_total = w.second_quarter_total + 50 := by sorry
  have h6 : w.third_quarter_female = w.second_quarter_female + 50 := by sorry
  have h7 : w.third_quarter_female = (62 : ℚ) / 100 * w.third_quarter_total := by sorry
  have h8 : w.final_total = w.third_quarter_total + 50 := by sorry
  have h9 : w.final_female = w.third_quarter_female + 10 := by sorry
  have h10 : w.final_female = (58 : ℚ) / 100 * w.final_total := by sorry
  sorry


end final_employee_count_l754_75417


namespace baker_cakes_theorem_l754_75477

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 145

/-- The number of cakes left after selling -/
def cakes_left : ℕ := 72

/-- The total number of cakes made by the baker -/
def total_cakes : ℕ := cakes_sold + cakes_left

theorem baker_cakes_theorem : total_cakes = 217 := by
  sorry

end baker_cakes_theorem_l754_75477


namespace max_x5_value_l754_75400

theorem max_x5_value (x1 x2 x3 x4 x5 : ℕ+) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) :
  x5 ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 ≤ a * b * c * d * 5 := by
  sorry

end max_x5_value_l754_75400


namespace right_triangle_properties_l754_75469

/-- A right triangle with hypotenuse 13 and one leg 5 -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : hypotenuse^2 = leg1^2 + leg2^2
  hypotenuse_is_13 : hypotenuse = 13
  leg1_is_5 : leg1 = 5

/-- Properties of the specific right triangle -/
theorem right_triangle_properties (t : RightTriangle) :
  t.leg2 = 12 ∧
  (1/2 : ℝ) * t.leg1 * t.leg2 = 30 ∧
  t.leg1 + t.leg2 + t.hypotenuse = 30 := by
  sorry

end right_triangle_properties_l754_75469


namespace fraction_equals_zero_l754_75489

theorem fraction_equals_zero (x : ℝ) : x = 5 → (x - 5) / (6 * x) = 0 := by
  sorry

end fraction_equals_zero_l754_75489


namespace two_zeros_iff_a_in_open_unit_interval_l754_75498

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - 3) - x + 2 * a

theorem two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (a > 0) →
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (0 < a ∧ a < 1) :=
sorry

end two_zeros_iff_a_in_open_unit_interval_l754_75498


namespace available_seats_l754_75432

theorem available_seats (total_seats : ℕ) (taken_fraction : ℚ) (broken_fraction : ℚ) : 
  total_seats = 500 →
  taken_fraction = 2 / 5 →
  broken_fraction = 1 / 10 →
  (total_seats : ℚ) - (taken_fraction * total_seats + broken_fraction * total_seats) = 250 := by
  sorry

end available_seats_l754_75432


namespace binomial_coefficient_and_increase_l754_75450

variable (n : ℕ)

theorem binomial_coefficient_and_increase :
  (Nat.choose n 2 = n * (n - 1) / 2) ∧
  (Nat.choose (n + 1) 2 - Nat.choose n 2 = n) := by
  sorry

end binomial_coefficient_and_increase_l754_75450


namespace f_50_solutions_l754_75448

-- Define f_0
def f_0 (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define f_n recursively
def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => f_0 x
  | n + 1 => |f n x| - 1

-- Theorem statement
theorem f_50_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f 50 x = 0) ∧ 
                    (∀ x ∉ S, f 50 x ≠ 0) ∧ 
                    Finset.card S = 4 :=
sorry

end f_50_solutions_l754_75448


namespace problem_solving_probability_l754_75484

theorem problem_solving_probability :
  let p_xavier : ℚ := 1/4
  let p_yvonne : ℚ := 2/3
  let p_zelda : ℚ := 5/8
  let p_william : ℚ := 7/10
  (p_xavier * p_yvonne * p_william * (1 - p_zelda) : ℚ) = 7/160 := by
  sorry

end problem_solving_probability_l754_75484


namespace major_premise_is_false_l754_75492

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem major_premise_is_false :
  ¬(∀ (l : Line) (p : Plane),
    parallel_line_plane l p →
    ∀ (m : Line), contained_in_plane m p → parallel_lines l m) :=
by sorry

end major_premise_is_false_l754_75492


namespace matching_shoes_probability_l754_75485

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 23 := by
  sorry

end matching_shoes_probability_l754_75485


namespace equation_solution_l754_75482

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x - 3 = 5 ∧ x = 112 :=
by sorry

end equation_solution_l754_75482


namespace three_digit_multiples_of_seven_l754_75488

theorem three_digit_multiples_of_seven : 
  (Finset.filter (fun k => 100 ≤ 7 * k ∧ 7 * k ≤ 999) (Finset.range 1000)).card = 128 := by
  sorry

end three_digit_multiples_of_seven_l754_75488


namespace quadratic_one_solution_l754_75490

theorem quadratic_one_solution (c : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + c = 0) ↔ c = 49 / 12 := by
  sorry

end quadratic_one_solution_l754_75490


namespace numbers_less_than_reciprocals_l754_75425

theorem numbers_less_than_reciprocals : ∃ (S : Set ℝ), 
  S = {-1/2, -3, 3, 1/2, 0} ∧ 
  (∀ x ∈ S, x ≠ 0 → (x < 1/x ↔ (x = -3 ∨ x = 1/2))) := by
  sorry

end numbers_less_than_reciprocals_l754_75425


namespace xiaoxia_exceeds_xiaoming_l754_75444

/-- Represents the savings of a person over time -/
structure Savings where
  initial : ℕ  -- Initial savings
  monthly : ℕ  -- Monthly savings rate
  months : ℕ   -- Number of months passed

/-- Calculates the total savings after a given number of months -/
def totalSavings (s : Savings) : ℕ :=
  s.initial + s.monthly * s.months

/-- Xiaoxia's savings parameters -/
def xiaoxia : Savings :=
  { initial := 52, monthly := 15, months := 0 }

/-- Xiaoming's savings parameters -/
def xiaoming : Savings :=
  { initial := 70, monthly := 12, months := 0 }

/-- Theorem stating when Xiaoxia's savings exceed Xiaoming's -/
theorem xiaoxia_exceeds_xiaoming (n : ℕ) :
  totalSavings { xiaoxia with months := n } > totalSavings { xiaoming with months := n } ↔
  52 + 15 * n > 70 + 12 * n :=
sorry

end xiaoxia_exceeds_xiaoming_l754_75444


namespace linear_function_composition_l754_75494

/-- Given two functions f and g, where f is a linear function with real coefficients a and b,
    and g is defined as g(x) = 3x - 4, prove that a + b = 11/3 if g(f(x)) = 4x + 3 for all x. -/
theorem linear_function_composition (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4) = 4 * x + 3) →
  a + b = 11 / 3 := by
  sorry

end linear_function_composition_l754_75494


namespace lee_cookies_with_five_cups_l754_75475

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies_with_five_cups :
  cookies_per_cups 5 = 40 := by
  sorry

end lee_cookies_with_five_cups_l754_75475


namespace exists_column_with_n_colors_l754_75404

/-- Represents a color in the grid -/
structure Color where
  id : Nat

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
structure Grid where
  size : Nat
  cells : List Cell

/-- Checks if a subgrid contains all n^2 colors -/
def containsAllColors (g : Grid) (n : Nat) (startRow : Nat) (startCol : Nat) : Prop :=
  ∀ c : Color, ∃ cell ∈ g.cells, 
    cell.row ≥ startRow ∧ cell.row < startRow + n ∧
    cell.col ≥ startCol ∧ cell.col < startCol + n ∧
    cell.color = c

/-- Checks if a row contains n distinct colors -/
def rowHasNColors (g : Grid) (n : Nat) (row : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.row = row ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.row = row → cell.color ∈ colors)

/-- Checks if a column contains exactly n distinct colors -/
def columnHasExactlyNColors (g : Grid) (n : Nat) (col : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.col = col ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.col = col → cell.color ∈ colors)

/-- The main theorem -/
theorem exists_column_with_n_colors (g : Grid) (n : Nat) :
  (∃ m : Nat, g.size = m * n) →
  (∀ i j : Nat, i < g.size - n + 1 → j < g.size - n + 1 → containsAllColors g n i j) →
  (∃ row : Nat, row < g.size ∧ rowHasNColors g n row) →
  (∃ col : Nat, col < g.size ∧ columnHasExactlyNColors g n col) :=
by sorry

end exists_column_with_n_colors_l754_75404


namespace toy_store_revenue_ratio_l754_75414

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  J = (1/3) * N →
  D = 3.75 * ((N + J) / 2) →
  N / D = 2/5 := by
sorry

end toy_store_revenue_ratio_l754_75414


namespace chemistry_question_ratio_l754_75419

theorem chemistry_question_ratio 
  (total_multiple_choice : ℕ) 
  (total_problem_solving : ℕ) 
  (problem_solving_fraction_written : ℚ) 
  (remaining_questions : ℕ) : 
  total_multiple_choice = 35 →
  total_problem_solving = 15 →
  problem_solving_fraction_written = 1/3 →
  remaining_questions = 31 →
  (total_multiple_choice - remaining_questions + 
   (total_problem_solving - ⌊total_problem_solving * problem_solving_fraction_written⌋)) / 
   total_multiple_choice = 9/35 :=
by sorry

end chemistry_question_ratio_l754_75419


namespace largest_angle_in_tangent_circles_triangle_l754_75423

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis --/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- Theorem about the largest angle in a triangle formed by centers of three mutually tangent circles --/
theorem largest_angle_in_tangent_circles_triangle (A B C : Circle) :
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent C A ∧
  is_tangent_to_x_axis A ∧
  is_tangent_to_x_axis B ∧
  is_tangent_to_x_axis C →
  ∃ γ : ℝ, π/2 < γ ∧ γ ≤ 2 * Real.arcsin (4/5) ∧ 
  γ = max (Real.arccos ((A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 - A.radius^2 - C.radius^2) / (2 * A.radius * C.radius))
          (max (Real.arccos ((B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 - B.radius^2 - C.radius^2) / (2 * B.radius * C.radius))
               (Real.arccos ((A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 - A.radius^2 - B.radius^2) / (2 * A.radius * B.radius))) :=
by sorry

end largest_angle_in_tangent_circles_triangle_l754_75423


namespace circle_center_transformation_l754_75451

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point up by a given amount -/
def translate_up (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- The final position of the center of circle S after transformations -/
def final_position (initial : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_x (reflect_y initial)) 5

theorem circle_center_transformation :
  final_position (3, -4) = (-3, 9) := by
  sorry

end circle_center_transformation_l754_75451


namespace chess_tournament_games_per_pair_l754_75408

/-- Represents a chess tournament with a given number of players and total games. -/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ

/-- Calculates the number of times each player plays against each opponent in a chess tournament. -/
def games_per_pair (tournament : ChessTournament) : ℚ :=
  (2 * tournament.total_games : ℚ) / (tournament.num_players * (tournament.num_players - 1))

/-- Theorem stating that in a chess tournament with 18 players and 306 total games,
    each player plays against each opponent exactly 2 times. -/
theorem chess_tournament_games_per_pair :
  let tournament := ChessTournament.mk 18 306
  games_per_pair tournament = 2 := by
  sorry


end chess_tournament_games_per_pair_l754_75408


namespace chinese_chess_draw_probability_l754_75409

/-- The probability of A and B drawing in Chinese chess -/
theorem chinese_chess_draw_probability 
  (p_a_not_lose : ℝ) 
  (p_b_not_lose : ℝ) 
  (h1 : p_a_not_lose = 0.8) 
  (h2 : p_b_not_lose = 0.7) : 
  ∃ (p_draw : ℝ), p_draw = 0.5 ∧ p_a_not_lose = (1 - p_b_not_lose) + p_draw :=
sorry

end chinese_chess_draw_probability_l754_75409


namespace child_admission_is_five_l754_75470

/-- Calculates the admission price for children given the following conditions:
  * Adult admission is $8
  * Total amount paid is $201
  * Total number of tickets is 33
  * Number of children's tickets is 21
-/
def childAdmissionPrice (adultPrice totalPaid totalTickets childTickets : ℕ) : ℕ :=
  (totalPaid - adultPrice * (totalTickets - childTickets)) / childTickets

/-- Proves that the admission price for children is $5 under the given conditions -/
theorem child_admission_is_five :
  childAdmissionPrice 8 201 33 21 = 5 := by
  sorry

end child_admission_is_five_l754_75470


namespace shaded_area_problem_l754_75436

theorem shaded_area_problem (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 4 →
  triangle_height = 3 →
  square_side * square_side - (1 / 2 * triangle_base * triangle_height) = 10 := by
  sorry

end shaded_area_problem_l754_75436


namespace square_with_specific_digits_l754_75479

theorem square_with_specific_digits (S : ℕ) : 
  (∃ (E : ℕ), S^2 = 10 * (10 * (10^100 * E + 2*E) + 2) + 5) →
  (S^2 % 10 = 5 ∧ S = (10^101 + 5) / 3) :=
by sorry

end square_with_specific_digits_l754_75479


namespace geometric_sequence_sum_constant_l754_75429

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3^(n-2) + k, prove that k = -1/9 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  (∀ n : ℕ, S n = 3^(n - 2) + k) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n - 1)) →
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = a (n + 1) / a n) →
  k = -1/9 := by
  sorry

end geometric_sequence_sum_constant_l754_75429


namespace present_worth_calculation_l754_75478

/-- Calculates the present worth given the banker's gain, interest rate, and time period -/
def present_worth (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  bankers_gain / (interest_rate * time)

/-- Theorem stating that under given conditions, the present worth is 120 -/
theorem present_worth_calculation :
  let bankers_gain : ℚ := 24
  let interest_rate : ℚ := 1/10  -- 10% as a rational number
  let time : ℚ := 2
  present_worth bankers_gain interest_rate time = 120 := by
sorry

#eval present_worth 24 (1/10) 2

end present_worth_calculation_l754_75478


namespace circumradius_inscribed_radius_inequality_l754_75428

/-- A triangle with its circumscribed and inscribed circles -/
structure Triangle where
  -- Radius of the circumscribed circle
  R : ℝ
  -- Radius of the inscribed circle
  r : ℝ
  -- Predicate indicating if the triangle is equilateral
  is_equilateral : Prop

/-- The radius of the circumscribed circle is at least twice the radius of the inscribed circle,
    with equality if and only if the triangle is equilateral -/
theorem circumradius_inscribed_radius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ t.is_equilateral) := by
  sorry

end circumradius_inscribed_radius_inequality_l754_75428


namespace max_value_of_f_l754_75422

def f (x : ℝ) := x^4 - 4*x + 3

theorem max_value_of_f : 
  ∃ (m : ℝ), m = 72 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ m :=
by sorry

end max_value_of_f_l754_75422


namespace quadratic_triple_root_l754_75437

/-- For a quadratic equation ax^2 + bx + c = 0, one root is triple the other 
    if and only if 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) ↔ 
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_triple_root_l754_75437


namespace fraction_equality_solution_l754_75462

theorem fraction_equality_solution : ∃! x : ℚ, (1 + x) / (5 + x) = 1 / 3 := by
  sorry

end fraction_equality_solution_l754_75462


namespace four_color_theorem_l754_75481

/-- A type representing the four colors used for edge coloring -/
inductive EdgeColor
| one
| two
| three
| four

/-- A graph with edges colored using four colors -/
structure ColoredGraph (α : Type*) where
  edges : α → α → Option EdgeColor
  edge_coloring_property : ∀ (a b c : α), 
    edges a b ≠ none → edges b c ≠ none → 
    ∀ (d : α), edges c d ≠ none → 
    edges a b ≠ edges c d

/-- A type representing the four colors used for vertex coloring -/
inductive VertexColor
| one
| two
| three
| four

/-- A proper vertex coloring of a graph -/
def ProperVertexColoring (G : ColoredGraph α) (f : α → VertexColor) :=
  ∀ (a b : α), G.edges a b ≠ none → f a ≠ f b

theorem four_color_theorem (α : Type*) (G : ColoredGraph α) :
  ∃ (f : α → VertexColor), ProperVertexColoring G f :=
sorry

end four_color_theorem_l754_75481


namespace concert_problem_l754_75463

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  laura : ℕ

/-- Conditions for the concert problem -/
def ConcertConditions (sc : SongCount) : Prop :=
  sc.hanna = 9 ∧
  sc.mary = 3 ∧
  sc.alina > sc.mary ∧ sc.alina < sc.hanna ∧
  sc.tina > sc.mary ∧ sc.tina < sc.hanna ∧
  sc.laura > sc.mary ∧ sc.laura < sc.hanna

/-- The total number of songs performed -/
def TotalSongs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna + sc.laura) / 4

/-- Theorem stating that under the given conditions, the total number of songs is 9 -/
theorem concert_problem (sc : SongCount) :
  ConcertConditions sc → TotalSongs sc = 9 := by
  sorry


end concert_problem_l754_75463


namespace opposite_of_three_l754_75499

/-- The opposite of a real number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of 3 is -3. -/
theorem opposite_of_three : opposite 3 = -3 := by
  sorry

end opposite_of_three_l754_75499


namespace distance_between_cities_l754_75486

/-- The distance between two cities given specific travel conditions -/
theorem distance_between_cities (cara_speed dan_min_speed : ℝ) 
  (dan_delay : ℝ) (h1 : cara_speed = 30) (h2 : dan_min_speed = 36) 
  (h3 : dan_delay = 1) : 
  ∃ D : ℝ, D = 180 ∧ D / dan_min_speed = D / cara_speed - dan_delay := by
  sorry

end distance_between_cities_l754_75486


namespace alans_current_rate_prove_alans_current_rate_l754_75445

/-- Alan's attempt to beat Kevin's hot wings eating record -/
theorem alans_current_rate (kevin_wings : ℕ) (kevin_time : ℕ) (alan_additional : ℕ) : ℕ :=
  let kevin_rate := kevin_wings / kevin_time
  let alan_target_rate := kevin_rate + 1
  alan_target_rate - alan_additional

/-- Proof of Alan's current rate of eating hot wings -/
theorem prove_alans_current_rate :
  alans_current_rate 64 8 4 = 4 := by
  sorry

end alans_current_rate_prove_alans_current_rate_l754_75445


namespace modulus_of_z_l754_75472

theorem modulus_of_z : Complex.abs ((1 - Complex.I) * Complex.I) = Real.sqrt 2 := by sorry

end modulus_of_z_l754_75472


namespace container_volume_ratio_l754_75480

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 → 
  (4/5 * A = 2/3 * B) → 
  (A / B = 5/6) := by
sorry

end container_volume_ratio_l754_75480


namespace actors_in_one_hour_show_l754_75460

/-- Calculates the number of actors in a show given the show duration, performance time per set, and number of actors per set. -/
def actors_in_show (show_duration : ℕ) (performance_time : ℕ) (actors_per_set : ℕ) : ℕ :=
  (show_duration / performance_time) * actors_per_set

/-- Proves that given the specified conditions, the number of actors in a 1-hour show is 20. -/
theorem actors_in_one_hour_show :
  actors_in_show 60 15 5 = 20 := by
  sorry

end actors_in_one_hour_show_l754_75460


namespace min_value_a_l754_75426

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / Real.exp x

theorem min_value_a :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f x ≤ a) ∧ 
  (∀ (b : ℝ), (∃ (x : ℝ), x ≥ -2 ∧ f x > b) → b < 1 - 1 / Real.exp 1) :=
sorry

end min_value_a_l754_75426


namespace circle_area_tripled_l754_75403

theorem circle_area_tripled (r n : ℝ) (h : r > 0) (h_n : n > 0) :
  π * (r + n)^2 = 3 * π * r^2 → r = n * (1 + Real.sqrt 3) / 2 := by
  sorry

end circle_area_tripled_l754_75403


namespace flowers_per_pot_l754_75401

theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 544) (h2 : total_flowers = 17408) :
  total_flowers / total_pots = 32 := by
  sorry

end flowers_per_pot_l754_75401


namespace willey_farm_land_allocation_l754_75491

/-- The Willey Farm Collective land allocation problem -/
theorem willey_farm_land_allocation :
  let corn_cost : ℝ := 42
  let wheat_cost : ℝ := 35
  let total_capital : ℝ := 165200
  let wheat_acres : ℝ := 3400
  let corn_acres : ℝ := (total_capital - wheat_cost * wheat_acres) / corn_cost
  corn_acres + wheat_acres = 4500 := by
  sorry

end willey_farm_land_allocation_l754_75491


namespace round_trip_average_speed_l754_75497

/-- The average speed of a round trip, given the outbound speed and the fact that the return journey takes twice as long -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h1 : outbound_speed = 54) 
  (h2 : return_time = 2 * outbound_time) : 
  average_speed = 36 := by
  sorry

#check round_trip_average_speed

end round_trip_average_speed_l754_75497


namespace gcd_of_specific_numbers_l754_75413

theorem gcd_of_specific_numbers : 
  let m : ℕ := 3333333
  let n : ℕ := 99999999
  Nat.gcd m n = 3 := by
sorry

end gcd_of_specific_numbers_l754_75413


namespace infinitely_many_primes_not_in_S_a_l754_75454

-- Define the set S_a
def S_a (a : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ b : ℕ, Odd b ∧ p ∣ (2^(2^a))^b - 1}

-- State the theorem
theorem infinitely_many_primes_not_in_S_a :
  ∀ a : ℕ, a > 0 → Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ S_a a} :=
by sorry

end infinitely_many_primes_not_in_S_a_l754_75454


namespace wire_division_l754_75411

/-- Given a wire that can be divided into two parts of 120 cm each with 2.4 cm left over,
    prove that when divided into three equal parts, each part is 80.8 cm long. -/
theorem wire_division (wire_length : ℝ) (h1 : wire_length = 2 * 120 + 2.4) :
  wire_length / 3 = 80.8 := by
sorry

end wire_division_l754_75411


namespace ball_height_properties_l754_75459

/-- The height of a ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating the maximum height and height at t = 1 -/
theorem ball_height_properties :
  (∀ t, h t ≤ 40) ∧ (h 1 = 40) := by
  sorry

end ball_height_properties_l754_75459


namespace haleys_stickers_haleys_stickers_specific_l754_75443

/-- Haley's sticker distribution problem -/
theorem haleys_stickers (num_friends : ℕ) (stickers_per_friend : ℕ) : 
  num_friends * stickers_per_friend = num_friends * stickers_per_friend := by
  sorry

/-- The specific case of Haley's problem -/
theorem haleys_stickers_specific : 9 * 8 = 72 := by
  sorry

end haleys_stickers_haleys_stickers_specific_l754_75443


namespace quadratic_expression_equality_l754_75431

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := by
  sorry

end quadratic_expression_equality_l754_75431


namespace stream_speed_l754_75441

/-- Proves that the speed of the stream is 8 kmph given the conditions -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 8 := by
sorry

end stream_speed_l754_75441


namespace special_sale_savings_l754_75420

/-- Given a special sale where 25 tickets can be purchased for the price of 21.5 tickets,
    prove that buying 50 tickets at this rate results in a 14% savings compared to the original price. -/
theorem special_sale_savings : ∀ (P : ℝ), P > 0 →
  let sale_price : ℝ := 21.5 * P / 25
  let original_price_50 : ℝ := 50 * P
  let sale_price_50 : ℝ := 50 * sale_price
  let savings : ℝ := original_price_50 - sale_price_50
  let savings_percentage : ℝ := savings / original_price_50 * 100
  savings_percentage = 14 := by
  sorry

end special_sale_savings_l754_75420


namespace triangle_problem_l754_75421

open Real

theorem triangle_problem (A B C : ℝ) (h : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A + B = 3 * C ∧
  2 * sin (A - C) = sin B →
  sin A = 3 * sqrt 10 / 10 ∧
  (∀ AB : ℝ, AB = 5 → h = 6 ∧ h * AB / 2 = sin C * AB * sin A / 2) :=
by sorry

end triangle_problem_l754_75421


namespace total_pools_count_l754_75465

/-- The number of stores operated by Pat's Pool Supply -/
def pool_supply_stores : ℕ := 4

/-- The number of stores operated by Pat's Ark & Athletic Wear -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of swimming pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 3

/-- The number of pools in one Pat's Ark & Athletic Wear store -/
def pools_per_ark_athletic : ℕ := 200

/-- The total number of swimming pools across all Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def total_pools : ℕ := pool_supply_stores * pool_ratio * pools_per_ark_athletic + ark_athletic_stores * pools_per_ark_athletic

theorem total_pools_count : total_pools = 3600 := by
  sorry

end total_pools_count_l754_75465


namespace distance_OQ_l754_75440

-- Define the geometric setup
structure GeometricSetup where
  R : ℝ  -- Radius of larger circle
  r : ℝ  -- Radius of smaller circle
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C

-- Define the theorem
theorem distance_OQ (setup : GeometricSetup) : 
  ∃ (OQ : ℝ), OQ = Real.sqrt (setup.R^2 - 2*setup.r*setup.R) :=
sorry

end distance_OQ_l754_75440


namespace circle_area_with_diameter_10_l754_75446

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l754_75446


namespace digit_divisible_by_9_l754_75416

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem digit_divisible_by_9 :
  is_divisible_by_9 5274 ∧ 
  ∀ B : ℕ, B ≤ 9 → B ≠ 4 → ¬(is_divisible_by_9 (5270 + B)) :=
by sorry

end digit_divisible_by_9_l754_75416


namespace school_children_count_l754_75433

theorem school_children_count :
  ∀ (total_children : ℕ) (total_bananas : ℕ),
    total_bananas = 2 * total_children →
    total_bananas = 4 * (total_children - 390) →
    total_children = 780 := by
  sorry

end school_children_count_l754_75433


namespace min_value_when_a_2_a_values_for_max_3_l754_75424

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1: Minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = -1 ∧ ∀ x ∈ Set.Icc 0 3, f 2 x ≥ min :=
sorry

-- Part 2: Values of a for maximum 3 in [0, 1]
theorem a_values_for_max_3 :
  (∃ (max : ℝ), max = 3 ∧ ∀ x ∈ Set.Icc 0 1, f a x ≤ max) →
  (a = -2 ∨ a = 3) :=
sorry

end min_value_when_a_2_a_values_for_max_3_l754_75424


namespace cycle_original_price_l754_75455

/-- The original price of a cycle sold at a loss -/
def original_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem: The original price of a cycle is 1750, given a selling price of 1610 and a loss of 8% -/
theorem cycle_original_price : 
  original_price 1610 8 = 1750 := by
  sorry

end cycle_original_price_l754_75455


namespace monomial_replacement_l754_75461

theorem monomial_replacement (x : ℝ) : 
  let expression := (x^4 - 3)^2 + (x^3 + 3*x)^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    expression = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end monomial_replacement_l754_75461


namespace sum_of_numbers_l754_75466

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end sum_of_numbers_l754_75466


namespace class_average_weight_l754_75493

/-- Given two sections A and B in a class, with their respective number of students and average weights,
    prove that the average weight of the whole class is as calculated. -/
theorem class_average_weight 
  (students_A : ℕ) (students_B : ℕ) 
  (avg_weight_A : ℝ) (avg_weight_B : ℝ) :
  students_A = 40 →
  students_B = 20 →
  avg_weight_A = 50 →
  avg_weight_B = 40 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 46.67 :=
by sorry

end class_average_weight_l754_75493


namespace roots_squared_relation_l754_75435

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem roots_squared_relation (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → p a b c (x^2) = 0) →
  a = -1 ∧ b = -2 ∧ c = 16 :=
by sorry

end roots_squared_relation_l754_75435


namespace divisibility_equivalence_l754_75476

def divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem divisibility_equivalence :
  (∀ n : ℕ, divisible n 6 → divisible n 3) ↔
  (∀ n : ℕ, ¬(divisible n 3) → ¬(divisible n 6)) ∧
  (∀ n : ℕ, ¬(divisible n 6) ∨ divisible n 3) :=
by sorry

end divisibility_equivalence_l754_75476


namespace functional_equation_solution_l754_75473

theorem functional_equation_solution (f : ℝ → ℝ) (h_continuous : Continuous f) 
  (h_equation : ∀ x y : ℝ, f (x + y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ c : ℝ, ∀ x : ℝ, f x = Real.exp (c * x)) := by
  sorry

end functional_equation_solution_l754_75473
