import Mathlib

namespace second_month_sale_l3336_333643

def sale_month1 : ℕ := 6235
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = desired_average * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end second_month_sale_l3336_333643


namespace pear_weight_proof_l3336_333623

/-- The weight of one pear in grams -/
def pear_weight : ℝ := 120

theorem pear_weight_proof :
  let apple_weight : ℝ := 530
  let apple_count : ℕ := 12
  let pear_count : ℕ := 8
  let weight_difference : ℝ := 5400
  apple_count * apple_weight = pear_count * pear_weight + weight_difference →
  pear_weight = 120 := by
sorry

end pear_weight_proof_l3336_333623


namespace golden_triangle_ratio_l3336_333608

theorem golden_triangle_ratio (t : ℝ) (h : t = (Real.sqrt 5 - 1) / 2) :
  (1 - 2 * Real.sin (27 * π / 180) ^ 2) / (2 * t * Real.sqrt (4 - t^2)) = 1/4 := by
  sorry

end golden_triangle_ratio_l3336_333608


namespace fridge_cost_difference_l3336_333602

theorem fridge_cost_difference (total_budget : ℕ) (tv_cost : ℕ) (computer_cost : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : computer_cost = 250)
  (h4 : ∃ fridge_cost : ℕ, fridge_cost > computer_cost ∧ 
        fridge_cost + tv_cost + computer_cost = total_budget) :
  ∃ fridge_cost : ℕ, fridge_cost - computer_cost = 500 := by
sorry

end fridge_cost_difference_l3336_333602


namespace acid_mixture_concentration_l3336_333683

/-- Proves that mixing 1.2 L of 10% acid solution with 0.8 L of 5% acid solution 
    results in a 2 L solution with 8% acid concentration -/
theorem acid_mixture_concentration : 
  let total_volume : Real := 2
  let volume_10_percent : Real := 1.2
  let volume_5_percent : Real := total_volume - volume_10_percent
  let concentration_10_percent : Real := 10 / 100
  let concentration_5_percent : Real := 5 / 100
  let total_acid : Real := 
    volume_10_percent * concentration_10_percent + 
    volume_5_percent * concentration_5_percent
  let final_concentration : Real := (total_acid / total_volume) * 100
  final_concentration = 8 := by sorry

end acid_mixture_concentration_l3336_333683


namespace ratio_squares_sum_l3336_333629

theorem ratio_squares_sum (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  c / b = 5 / 2 ∧ 
  b = 14 → 
  a^2 + b^2 + c^2 = 1862 := by
sorry

end ratio_squares_sum_l3336_333629


namespace distance_to_origin_of_complex_number_l3336_333626

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 :=
sorry

end distance_to_origin_of_complex_number_l3336_333626


namespace election_votes_theorem_l3336_333621

theorem election_votes_theorem (candidates : ℕ) (winner_percentage : ℝ) (majority : ℕ) 
  (h1 : candidates = 4)
  (h2 : winner_percentage = 0.7)
  (h3 : majority = 3000) :
  ∃ total_votes : ℕ, 
    (↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = majority) ∧ 
    total_votes = 7500 := by
  sorry

end election_votes_theorem_l3336_333621


namespace correct_average_l3336_333614

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum / numbers.card = 19 →
  incorrect_number = 26 →
  correct_number = 76 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 24 :=
by sorry

end correct_average_l3336_333614


namespace not_sum_of_two_squares_l3336_333687

theorem not_sum_of_two_squares (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ (a b : ℤ), n = a^2 + b^2 := by
  sorry

end not_sum_of_two_squares_l3336_333687


namespace range_of_a_l3336_333659

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 3 * (a - 3) * x^2 + 1 / x = 0) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → 3 * x^2 - 2 * a * x - 3 ≥ 0) → 
  a ∈ Set.Iic 0 := by
sorry

end range_of_a_l3336_333659


namespace element_in_two_pairs_l3336_333624

/-- A system of elements and pairs satisfying the given conditions -/
structure PairSystem (n : ℕ) where
  -- The set of elements
  elements : Fin n → Type
  -- The set of pairs
  pairs : Fin n → Set (Fin n)
  -- Two pairs share exactly one element iff they form a pair
  share_condition : ∀ i j : Fin n, 
    (∃! k : Fin n, k ∈ pairs i ∧ k ∈ pairs j) ↔ j ∈ pairs i

/-- Every element is in exactly two pairs -/
theorem element_in_two_pairs {n : ℕ} (sys : PairSystem n) :
  ∀ k : Fin n, ∃! (i j : Fin n), i ≠ j ∧ k ∈ sys.pairs i ∧ k ∈ sys.pairs j :=
sorry

end element_in_two_pairs_l3336_333624


namespace binomial_18_10_l3336_333654

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end binomial_18_10_l3336_333654


namespace parabola_chord_midpoint_trajectory_l3336_333672

/-- The equation of the trajectory of the midpoint of a chord passing through the focus of the parabola y² = 4x is y² = 2x - 2 -/
theorem parabola_chord_midpoint_trajectory (x y : ℝ) :
  (∀ x₀ y₀, y₀^2 = 4*x₀ → -- Parabola equation
   ∃ a b : ℝ, (y - y₀)^2 = 4*(a^2 + b^2)*(x - x₀) ∧ -- Chord passing through focus
   x = (x₀ + a)/2 ∧ y = (y₀ + b)/2) -- Midpoint of chord
  → y^2 = 2*x - 2 := by sorry

end parabola_chord_midpoint_trajectory_l3336_333672


namespace polynomial_never_33_l3336_333693

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
sorry

end polynomial_never_33_l3336_333693


namespace no_integer_solutions_l3336_333648

theorem no_integer_solutions (n k m l : ℕ) : 
  l ≥ 2 → 4 ≤ k → k ≤ n - 4 → Nat.choose n k ≠ m^l := by sorry

end no_integer_solutions_l3336_333648


namespace larger_integer_proof_l3336_333681

theorem larger_integer_proof (x y : ℕ) (h1 : y = 4 * x) (h2 : (x + 6) / y = 1 / 3) : y = 72 := by
  sorry

end larger_integer_proof_l3336_333681


namespace third_grade_sample_size_l3336_333676

/-- Calculates the number of students sampled from a specific grade in a stratified sampling. -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample : ℕ) : ℕ :=
  (grade_population * total_sample) / total_population

/-- Proves that in a stratified sampling of 40 students from a population of 1000 students,
    where 400 students are in the third grade, the number of students sampled from the third grade is 16. -/
theorem third_grade_sample_size :
  stratified_sample_size 1000 400 40 = 16 := by
  sorry

end third_grade_sample_size_l3336_333676


namespace smallest_four_digit_divisible_by_24_l3336_333669

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- A number is divisible by another number if the remainder of their division is zero -/
def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem smallest_four_digit_divisible_by_24 :
  is_four_digit 1104 ∧ 
  is_divisible_by 1104 24 ∧ 
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 24 → 1104 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_24_l3336_333669


namespace smallest_integer_satisfying_inequality_l3336_333644

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 2*x - 7 → x ≥ 8 ∧ 8 < 2*8 - 7 := by
  sorry

end smallest_integer_satisfying_inequality_l3336_333644


namespace cycle_selling_price_l3336_333698

theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 900 →
  gain_percentage = 22.22222222222222 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1100 := by
sorry

end cycle_selling_price_l3336_333698


namespace circle_graph_proportion_l3336_333689

theorem circle_graph_proportion (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) :
  (y = 3.6 * x) ↔ (y / 360 = x / 100) :=
by sorry

end circle_graph_proportion_l3336_333689


namespace pascal_row_10_sum_l3336_333625

/-- The sum of numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end pascal_row_10_sum_l3336_333625


namespace johns_total_distance_l3336_333673

/-- Calculates the total distance driven given the speed and time for each segment of a trip. -/
def total_distance (speed1 speed2 speed3 speed4 : ℝ) (time1 time2 time3 time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that John's total distance driven is 470 miles. -/
theorem johns_total_distance :
  total_distance 45 55 60 50 2 3 1.5 2.5 = 470 := by
  sorry

#eval total_distance 45 55 60 50 2 3 1.5 2.5

end johns_total_distance_l3336_333673


namespace expression_simplification_l3336_333664

theorem expression_simplification (a : ℝ) (h : a/2 - 2/a = 3) :
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 := by
  sorry

end expression_simplification_l3336_333664


namespace initial_students_count_l3336_333622

/-- The number of students at the start of the year. -/
def initial_students : ℕ := sorry

/-- The number of students who left during the year. -/
def students_left : ℕ := 5

/-- The number of new students who came during the year. -/
def new_students : ℕ := 8

/-- The number of students at the end of the year. -/
def final_students : ℕ := 11

/-- Theorem stating that the initial number of students is 8. -/
theorem initial_students_count :
  initial_students = final_students - (new_students - students_left) := by sorry

end initial_students_count_l3336_333622


namespace y_minimizer_l3336_333699

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3*x + 5

/-- The theorem stating that (2a + 2b - 3) / 4 minimizes y -/
theorem y_minimizer (a b : ℝ) :
  ∃ (x_min : ℝ), x_min = (2*a + 2*b - 3) / 4 ∧
    ∀ (x : ℝ), y x a b ≥ y x_min a b :=
by
  sorry

end y_minimizer_l3336_333699


namespace scalene_triangle_bisector_inequality_l3336_333642

/-- Given a scalene triangle with longest angle bisector l₁, shortest angle bisector l₂, and area S,
    prove that l₁² > √3 S > l₂². -/
theorem scalene_triangle_bisector_inequality (a b c : ℝ) (h_scalene : a > b ∧ b > c ∧ c > 0) :
  ∃ (l₁ l₂ S : ℝ), l₁ > 0 ∧ l₂ > 0 ∧ S > 0 ∧
  (∀ l : ℝ, (l > 0 ∧ l ≠ l₁ ∧ l ≠ l₂) → (l < l₁ ∧ l > l₂)) ∧
  S = (1/2) * b * c * Real.sin ((2/3) * Real.pi) ∧
  l₁^2 > Real.sqrt 3 * S ∧ Real.sqrt 3 * S > l₂^2 :=
by sorry

end scalene_triangle_bisector_inequality_l3336_333642


namespace vehicle_value_theorem_l3336_333639

def vehicle_value_last_year : ℝ := 20000

def depreciation_factor : ℝ := 0.8

def vehicle_value_this_year : ℝ := depreciation_factor * vehicle_value_last_year

theorem vehicle_value_theorem : vehicle_value_this_year = 16000 := by
  sorry

end vehicle_value_theorem_l3336_333639


namespace dianas_biking_distance_l3336_333613

/-- Diana's biking problem -/
theorem dianas_biking_distance
  (initial_speed : ℝ)
  (initial_time : ℝ)
  (tired_speed : ℝ)
  (total_time : ℝ)
  (h1 : initial_speed = 3)
  (h2 : initial_time = 2)
  (h3 : tired_speed = 1)
  (h4 : total_time = 6) :
  initial_speed * initial_time + tired_speed * (total_time - initial_time) = 10 :=
by sorry

end dianas_biking_distance_l3336_333613


namespace horner_rule_example_l3336_333667

def horner_polynomial (x : ℝ) : ℝ :=
  (((((x + 2) * x) * x - 3) * x + 7) * x - 2)

theorem horner_rule_example :
  horner_polynomial 2 = 64 := by
  sorry

end horner_rule_example_l3336_333667


namespace car_selling_problem_l3336_333632

/-- Calculates the net amount Chris receives from each buyer's offer --/
def net_amount (asking_price : ℝ) (inspection_cost : ℝ) (headlight_cost : ℝ) 
  (tire_cost : ℝ) (battery_cost : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) : ℝ × ℝ × ℝ :=
  let first_offer := asking_price - inspection_cost
  let second_offer := asking_price - (headlight_cost + tire_cost + battery_cost)
  let discounted_price := asking_price * (1 - discount_rate)
  let third_offer := discounted_price - (discounted_price * paint_job_rate)
  (first_offer, second_offer, third_offer)

/-- Theorem statement for the car selling problem --/
theorem car_selling_problem (asking_price : ℝ) (inspection_rate : ℝ) (headlight_cost : ℝ) 
  (tire_rate : ℝ) (battery_rate : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) :
  asking_price = 5200 ∧
  inspection_rate = 1/10 ∧
  headlight_cost = 80 ∧
  tire_rate = 3 ∧
  battery_rate = 2 ∧
  discount_rate = 15/100 ∧
  paint_job_rate = 1/5 →
  let (first, second, third) := net_amount asking_price (asking_price * inspection_rate) 
    headlight_cost (headlight_cost * tire_rate) (headlight_cost * tire_rate * battery_rate) 
    discount_rate paint_job_rate
  max first (max second third) - min first (min second third) = 1144 := by
  sorry


end car_selling_problem_l3336_333632


namespace factor_polynomial_l3336_333668

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 250 * x^13 = 25 * x^7 * (3 - 10 * x^6) := by
  sorry

end factor_polynomial_l3336_333668


namespace simplify_cube_roots_l3336_333628

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end simplify_cube_roots_l3336_333628


namespace special_function_at_one_l3336_333649

/-- A function satisfying certain properties on positive real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f (1 / x) = x * f x) ∧
  (∀ x > 0, ∀ y > 0, f x + f y = x + y + f (x * y))

/-- The value of f(1) for a function satisfying the special properties -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end special_function_at_one_l3336_333649


namespace board_numbers_transformation_impossibility_of_returning_to_original_numbers_l3336_333645

theorem board_numbers_transformation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a - b / 2) ^ 2 + (b + a / 2) ^ 2 > a ^ 2 + b ^ 2 := by
  sorry

theorem impossibility_of_returning_to_original_numbers :
  ∀ (numbers : List ℝ), 
  (∀ n ∈ numbers, n ≠ 0) →
  ∃ (new_numbers : List ℝ),
  (new_numbers.length = numbers.length) ∧
  (List.sum (List.map (λ x => x^2) new_numbers) > List.sum (List.map (λ x => x^2) numbers)) := by
  sorry

end board_numbers_transformation_impossibility_of_returning_to_original_numbers_l3336_333645


namespace ellipse_properties_l3336_333662

-- Define the ellipse and its properties
def Ellipse (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = b^2 + c^2

-- Define the points and conditions
def EllipseConditions (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) : Prop :=
  Ellipse a b c ∧
  F₁ = (-c, 0) ∧
  F₂ = (c, 0) ∧
  E = (a^2/c, 0) ∧
  (∃ k : ℝ, A.2 = k * (A.1 - a^2/c) ∧ B.2 = k * (B.1 - a^2/c)) ∧
  (∃ t : ℝ, A.1 - F₁.1 = t * (B.1 - F₂.1) ∧ A.2 - F₁.2 = t * (B.2 - F₂.2)) ∧
  (A.1 - F₁.1)^2 + (A.2 - F₁.2)^2 = 4 * ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2)

-- Theorem statement
theorem ellipse_properties (a b c : ℝ) (A B E F₁ F₂ : ℝ × ℝ) 
  (h : EllipseConditions a b c A B E F₁ F₂) :
  c / a = Real.sqrt 3 / 3 ∧ 
  (∃ k : ℝ, (A.2 - B.2) / (A.1 - B.1) = k ∧ (k = Real.sqrt 2 / 3 ∨ k = -Real.sqrt 2 / 3)) :=
sorry

end ellipse_properties_l3336_333662


namespace fraction_simplification_l3336_333692

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (3*x + 4) / 3 = (18*x + 7) / 12 := by
  sorry

end fraction_simplification_l3336_333692


namespace cards_per_student_l3336_333619

/-- Given that Joseph had 357 cards initially, has 15 students, and had 12 cards left after distribution,
    prove that the number of cards given to each student is 23. -/
theorem cards_per_student (total_cards : Nat) (num_students : Nat) (remaining_cards : Nat)
    (h1 : total_cards = 357)
    (h2 : num_students = 15)
    (h3 : remaining_cards = 12) :
    (total_cards - remaining_cards) / num_students = 23 :=
by sorry

end cards_per_student_l3336_333619


namespace initial_value_proof_l3336_333636

theorem initial_value_proof : 
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) → 
  (∃! x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0) ∧
  (∃ x : ℕ, (x + 294) % 456 = 0 ∧ ∀ y : ℕ, y < x → (y + 294) % 456 ≠ 0 ∧ x = 162) :=
by sorry

end initial_value_proof_l3336_333636


namespace circle_kinetic_energy_l3336_333611

/-- 
Given a circle with radius R and a point P on its diameter AB, where PC is a semicord perpendicular to AB,
if three unit masses move along PA, PB, and PC with constant velocities reaching A, B, and C respectively in one unit of time,
and the total kinetic energy expended is a^2 units, then:
1. The distance of P from A is R ± √(2a^2 - 3R^2)
2. The value of a^2 must satisfy 3/2 * R^2 ≤ a^2 < 2R^2
-/
theorem circle_kinetic_energy (R a : ℝ) (h : R > 0) :
  let PA : ℝ → ℝ := λ x => x
  let PB : ℝ → ℝ := λ x => 2 * R - x
  let PC : ℝ → ℝ := λ x => Real.sqrt (x * (2 * R - x))
  let kinetic_energy : ℝ → ℝ := λ x => (PA x)^2 / 2 + (PB x)^2 / 2 + (PC x)^2 / 2
  ∃ x : ℝ, 0 < x ∧ x < 2 * R ∧ kinetic_energy x = a^2 →
    (x = R + Real.sqrt (2 * a^2 - 3 * R^2) ∨ x = R - Real.sqrt (2 * a^2 - 3 * R^2)) ∧
    3 / 2 * R^2 ≤ a^2 ∧ a^2 < 2 * R^2 :=
by sorry

end circle_kinetic_energy_l3336_333611


namespace circle_equation_correct_l3336_333686

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.r = 1 ∧ c.center.θ = π/4 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔ (p.r * Real.cos p.θ - c.center.r)^2 + (p.r * Real.sin p.θ - c.center.r)^2 = c.radius^2 :=
sorry

end circle_equation_correct_l3336_333686


namespace sqrt_eight_div_sqrt_two_equals_two_l3336_333630

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end sqrt_eight_div_sqrt_two_equals_two_l3336_333630


namespace six_ronna_scientific_notation_l3336_333663

/-- Represents the number of zeros after a number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to its scientific notation -/
def ronna_to_scientific (n : ℕ) : ℝ := n * (10 : ℝ) ^ ronna_zeros

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 : ℝ) ^ 27 := by
  sorry

end six_ronna_scientific_notation_l3336_333663


namespace handkerchief_usage_per_day_l3336_333675

/-- Proves that given square handkerchiefs of 25 cm × 25 cm and total fabric usage of 3 m² over 8 days, 
    the number of handkerchiefs used per day is 6. -/
theorem handkerchief_usage_per_day 
  (handkerchief_side : ℝ) 
  (total_fabric_area : ℝ) 
  (days : ℕ) 
  (h1 : handkerchief_side = 25) 
  (h2 : total_fabric_area = 3) 
  (h3 : days = 8) : 
  (total_fabric_area * 10000) / (handkerchief_side ^ 2 * days) = 6 :=
by
  sorry

end handkerchief_usage_per_day_l3336_333675


namespace park_area_is_102400_l3336_333603

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length = 4 * breadth

/-- Calculates the perimeter of the park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Calculates the area of the park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Theorem: The area of the park is 102400 square meters -/
theorem park_area_is_102400 (park : RectangularPark) 
    (h_perimeter : perimeter park = 12 * 8 / 60 * 1000) : 
    area park = 102400 := by
  sorry


end park_area_is_102400_l3336_333603


namespace brick_width_correct_l3336_333691

/-- The width of a brick used to build a wall with given dimensions -/
def brick_width : ℝ :=
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  11.25

/-- Theorem stating that the calculated brick width is correct -/
theorem brick_width_correct :
  let wall_length : ℝ := 800  -- 8 m in cm
  let wall_height : ℝ := 600  -- 6 m in cm
  let wall_thickness : ℝ := 22.5
  let brick_count : ℕ := 1600
  let brick_length : ℝ := 100
  let brick_height : ℝ := 6
  wall_length * wall_height * wall_thickness = 
    brick_count * (brick_length * brick_width * brick_height) :=
by
  sorry

#eval brick_width

end brick_width_correct_l3336_333691


namespace horner_method_multiplications_l3336_333658

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Number of multiplications in Horner's method -/
def horner_multiplications (coeffs : List ℝ) : ℕ :=
  coeffs.length - 1

/-- The polynomial f(x) = 3x^4 + 3x^3 + 2x^2 + 6x + 1 -/
def f_coeffs : List ℝ := [3, 3, 2, 6, 1]

theorem horner_method_multiplications :
  horner_multiplications f_coeffs = 4 :=
by
  sorry

#eval horner_eval f_coeffs 0.5
#eval horner_multiplications f_coeffs

end horner_method_multiplications_l3336_333658


namespace fraction_division_problem_l3336_333610

theorem fraction_division_problem : (3/7 + 1/3) / (2/5) = 40/21 := by
  sorry

end fraction_division_problem_l3336_333610


namespace work_completion_l3336_333635

/-- Given a piece of work that requires 400 man-days to complete,
    prove that if it takes 26.666666666666668 days for a group of men to complete,
    then the number of men in that group is 15. -/
theorem work_completion (total_man_days : ℝ) (days_to_complete : ℝ) (num_men : ℝ) :
  total_man_days = 400 →
  days_to_complete = 26.666666666666668 →
  num_men * days_to_complete = total_man_days →
  num_men = 15 := by
sorry

end work_completion_l3336_333635


namespace candle_height_at_half_time_l3336_333670

/-- Calculates the total burning time for a candle of given height -/
def totalBurningTime (height : ℕ) : ℕ :=
  10 * (height * (height + 1) * (2 * height + 1)) / 6

/-- Calculates the height of the candle after a given time -/
def heightAfterTime (initialHeight : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialHeight - (Finset.filter (fun k => 10 * k * k ≤ elapsedTime) (Finset.range initialHeight)).card

theorem candle_height_at_half_time (initialHeight : ℕ) (halfTimeHeight : ℕ) :
  initialHeight = 150 →
  halfTimeHeight = heightAfterTime initialHeight (totalBurningTime initialHeight / 2) →
  halfTimeHeight = 80 := by
  sorry

#eval heightAfterTime 150 (totalBurningTime 150 / 2)

end candle_height_at_half_time_l3336_333670


namespace telescope_visual_range_increase_l3336_333688

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 50)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 200 := by
  sorry

end telescope_visual_range_increase_l3336_333688


namespace solution_implies_m_minus_n_equals_negative_three_l3336_333655

theorem solution_implies_m_minus_n_equals_negative_three :
  ∀ m n : ℤ,
  (3 * (-2) + 2 * 1 = m) →
  (n * (-2) - 1 = 1) →
  m - n = -3 := by
  sorry

end solution_implies_m_minus_n_equals_negative_three_l3336_333655


namespace bus_speed_with_stops_l3336_333685

/-- The speed of a bus including stoppages, given its speed excluding stoppages and stop time -/
theorem bus_speed_with_stops (speed_without_stops : ℝ) (stop_time : ℝ) :
  speed_without_stops = 54 →
  stop_time = 20 →
  let speed_with_stops := speed_without_stops * (60 - stop_time) / 60
  speed_with_stops = 36 := by
  sorry

#check bus_speed_with_stops

end bus_speed_with_stops_l3336_333685


namespace F_r_properties_l3336_333612

/-- Represents a point in the cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the polygon F_r -/
def F_r (r : ℝ) : Set Point :=
  {p : Point | p.x^2 + p.y^2 = r^2 ∧ (p.x * p.y)^2 = 1}

/-- The area of the polygon F_r as a function of r -/
noncomputable def area (r : ℝ) : ℝ :=
  sorry

/-- Predicate to check if a polygon is regular -/
def is_regular (s : Set Point) : Prop :=
  sorry

theorem F_r_properties :
  ∃ (A : ℝ → ℝ),
    (∀ r, A r = area r) ∧
    is_regular (F_r 1) ∧
    ∀ r > 1, is_regular (F_r r) := by
  sorry

end F_r_properties_l3336_333612


namespace fraction_transformation_l3336_333666

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : -a / (a - b) = a / (b - a) := by
  sorry

end fraction_transformation_l3336_333666


namespace point_in_second_quadrant_l3336_333680

theorem point_in_second_quadrant (A B : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end point_in_second_quadrant_l3336_333680


namespace yoe_speed_calculation_l3336_333641

/-- Yoe's speed in miles per hour -/
def yoe_speed : ℝ := 40

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Initial distance between Teena and Yoe in miles (Teena behind) -/
def initial_distance : ℝ := 7.5

/-- Time elapsed in hours -/
def time_elapsed : ℝ := 1.5

/-- Final distance between Teena and Yoe in miles (Teena ahead) -/
def final_distance : ℝ := 15

theorem yoe_speed_calculation : 
  yoe_speed = (teena_speed * time_elapsed - initial_distance - final_distance) / time_elapsed :=
by sorry

end yoe_speed_calculation_l3336_333641


namespace equation_solution_l3336_333620

theorem equation_solution (x : ℝ) :
  (x^2 - 7*x + 6)/(x - 1) + (2*x^2 + 7*x - 6)/(2*x - 1) = 1 ∧ 
  x ≠ 1 ∧ 
  x ≠ 1/2 →
  x = 1/2 := by
sorry

end equation_solution_l3336_333620


namespace lcm_24_30_40_l3336_333650

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by
  sorry

end lcm_24_30_40_l3336_333650


namespace three_true_propositions_l3336_333697

-- Definition of reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of equal area triangles
def equal_area_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of real solutions for quadratic equation
def has_real_solutions (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_area_triangles t1 t2)) ∧
  (∀ m : ℝ, ¬(has_real_solutions m) → m > 1) :=
sorry

end three_true_propositions_l3336_333697


namespace vector_at_t_6_l3336_333690

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem vector_at_t_6 (h0 : line_vector 0 = (2, -1, 3))
                      (h4 : line_vector 4 = (6, 7, -1)) :
  line_vector 6 = (8, 11, -3) := by sorry

end vector_at_t_6_l3336_333690


namespace geometric_sequence_sum_l3336_333605

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (a 1 + a 2 = 16) →                        -- first given condition
  (a 3 + a 4 = 24) →                        -- second given condition
  (a 7 + a 8 = 54) :=                       -- conclusion to prove
by sorry

end geometric_sequence_sum_l3336_333605


namespace impossibleToReachOpposite_l3336_333695

/-- Represents the color of a point -/
inductive Color
| White
| Black

/-- Represents a point on the circle -/
structure Point where
  position : Fin 2022
  color : Color

/-- The type of operation that can be performed -/
inductive Operation
| FlipAdjacent (i : Fin 2022)
| FlipWithGap (i : Fin 2022)

/-- The configuration of all points on the circle -/
def Configuration := Fin 2022 → Color

/-- Apply an operation to a configuration -/
def applyOperation (config : Configuration) (op : Operation) : Configuration :=
  sorry

/-- The initial configuration with one black point and others white -/
def initialConfig : Configuration :=
  sorry

/-- Check if a configuration is the opposite of the initial configuration -/
def isOppositeConfig (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to reach the opposite configuration -/
theorem impossibleToReachOpposite : 
  ∀ (ops : List Operation), 
    ¬(isOppositeConfig (ops.foldl applyOperation initialConfig)) :=
  sorry

end impossibleToReachOpposite_l3336_333695


namespace wall_length_calculation_l3336_333646

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the wall's length is approximately 86 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side) * 2 = wall_width * (round ((mirror_side * mirror_side) * 2 / wall_width)) :=
by
  sorry

end wall_length_calculation_l3336_333646


namespace garden_area_increase_l3336_333656

/-- Given a rectangular garden with dimensions 60 feet by 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangle_length : ℝ := 60
  let rectangle_width : ℝ := 20
  let rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)
  let square_side : ℝ := rectangle_perimeter / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let square_area : ℝ := square_side ^ 2
  square_area - rectangle_area = 400 := by
sorry

end garden_area_increase_l3336_333656


namespace complex_point_on_line_l3336_333671

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x - y + 1 = 0) → a = -1 := by
  sorry

end complex_point_on_line_l3336_333671


namespace line_intersection_x_axis_l3336_333637

/-- A line passing through two points intersects the x-axis at a specific point -/
theorem line_intersection_x_axis (x₁ y₁ x₂ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let x_intercept := b / m
  (x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6) →
  x_intercept = 10 ∧ m * x_intercept + b = 0 :=
by
  sorry

#check line_intersection_x_axis

end line_intersection_x_axis_l3336_333637


namespace at_least_one_negative_l3336_333627

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by sorry

end at_least_one_negative_l3336_333627


namespace complex_power_magnitude_l3336_333652

theorem complex_power_magnitude : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 2) ^ 6) = 1728 := by
  sorry

end complex_power_magnitude_l3336_333652


namespace expansion_and_reduction_l3336_333606

theorem expansion_and_reduction : 
  (234 * 205 = 47970) ∧ (86400 / 300 = 288) := by
  sorry

end expansion_and_reduction_l3336_333606


namespace league_games_and_weeks_l3336_333609

/-- Represents a sports league --/
structure League where
  num_teams : ℕ
  games_per_week : ℕ

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (league : League) : ℕ :=
  league.num_teams * (league.num_teams - 1) / 2

/-- Calculates the minimum number of weeks required to complete all games --/
def min_weeks (league : League) : ℕ :=
  (total_games league + league.games_per_week - 1) / league.games_per_week

/-- Theorem about the number of games and weeks in a specific league --/
theorem league_games_and_weeks :
  let league := League.mk 15 7
  total_games league = 105 ∧ min_weeks league = 15 := by
  sorry


end league_games_and_weeks_l3336_333609


namespace range_of_a_l3336_333604

/-- The solution set to the inequality a^2 - 4a + 3 < 0 -/
def P (a : ℝ) : Prop := a^2 - 4*a + 3 < 0

/-- The real number a for which (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all real numbers x -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0

/-- Given P ∨ Q is true, the range of values for the real number a is -2 < a < 3 -/
theorem range_of_a (a : ℝ) (h : P a ∨ Q a) : -2 < a ∧ a < 3 := by
  sorry

end range_of_a_l3336_333604


namespace temperature_difference_l3336_333618

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 12) (h2 : lowest = -1) :
  highest - lowest = 13 := by sorry

end temperature_difference_l3336_333618


namespace kangaroo_problem_l3336_333607

/-- Represents the number of days required to reach a target number of kangaroos -/
def daysToReach (initial : ℕ) (daily : ℕ) (target : ℕ) : ℕ :=
  if initial ≥ target then 0
  else ((target - initial) + (daily - 1)) / daily

theorem kangaroo_problem :
  let kameronKangaroos : ℕ := 100
  let bertInitial : ℕ := 20
  let bertDaily : ℕ := 2
  let christinaInitial : ℕ := 45
  let christinaDaily : ℕ := 3
  let davidInitial : ℕ := 10
  let davidDaily : ℕ := 5
  
  max (daysToReach bertInitial bertDaily kameronKangaroos)
      (max (daysToReach christinaInitial christinaDaily kameronKangaroos)
           (daysToReach davidInitial davidDaily kameronKangaroos)) = 40 := by
  sorry

end kangaroo_problem_l3336_333607


namespace brendas_age_l3336_333674

theorem brendas_age (addison janet brenda : ℕ) 
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 9)
  (h3 : addison = janet) : 
  brenda = 3 := by
  sorry

end brendas_age_l3336_333674


namespace dm_length_l3336_333647

/-- A square with side length 3 and two points that divide it into three equal areas -/
structure EqualAreaSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The point M on side AD -/
  m : ℝ
  /-- The point N on side AB -/
  n : ℝ
  /-- The side length is 3 -/
  side_eq : side = 3
  /-- The point M is between 0 and the side length -/
  m_range : 0 ≤ m ∧ m ≤ side
  /-- The point N is between 0 and the side length -/
  n_range : 0 ≤ n ∧ n ≤ side
  /-- CM and CN divide the square into three equal areas -/
  equal_areas : (1/2 * m * side) = (1/2 * n * side) ∧ (1/2 * m * side) = (1/3 * side^2)

/-- The length of DM in an EqualAreaSquare is 2 -/
theorem dm_length (s : EqualAreaSquare) : s.m = 2 := by
  sorry

end dm_length_l3336_333647


namespace angle_complement_supplement_l3336_333633

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 4 * (180 - x) → x = 60 := by
  sorry

end angle_complement_supplement_l3336_333633


namespace factorization_theorem_l3336_333677

theorem factorization_theorem (m n : ℝ) : 2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) := by
  sorry

end factorization_theorem_l3336_333677


namespace common_root_condition_l3336_333640

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) := by
  sorry

end common_root_condition_l3336_333640


namespace tangent_circle_equation_l3336_333694

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define point H
def H : ℝ × ℝ := (1, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 25/4

-- Theorem statement
theorem tangent_circle_equation :
  ∃ (A B : ℝ × ℝ),
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (∃ (m₁ m₂ : ℝ),
      (A.2 - H.2 = m₁ * (A.1 - H.1)) ∧
      (B.2 - H.2 = m₂ * (B.1 - H.1)) ∧
      (∀ (x y : ℝ), parabola x y → m₁ * (x - A.1) + A.2 ≥ y) ∧
      (∀ (x y : ℝ), parabola x y → m₂ * (x - B.1) + B.2 ≥ y)) →
    ∀ (x y : ℝ), circle_equation x y ↔ 
      ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2 :=
by sorry

end tangent_circle_equation_l3336_333694


namespace product_of_smallest_primes_l3336_333634

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes :
  (smallest_one_digit_primes.prod) * smallest_two_digit_prime = 66 := by
  sorry

end product_of_smallest_primes_l3336_333634


namespace paint_calculation_l3336_333631

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 * total_paint + 1/3 * (3/4 * total_paint) = 180) → total_paint = 360 := by
  sorry

end paint_calculation_l3336_333631


namespace systematic_sampling_50_5_l3336_333684

/-- Represents a list of product numbers selected using systematic sampling. -/
def systematicSample (totalProducts : ℕ) (sampleSize : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that systematic sampling of 5 products from 50 products
    results in the selection of products numbered 10, 20, 30, 40, and 50. -/
theorem systematic_sampling_50_5 :
  systematicSample 50 5 = [10, 20, 30, 40, 50] := by
  sorry

end systematic_sampling_50_5_l3336_333684


namespace arithmetic_sequence_tangent_l3336_333601

/-- Given an arithmetic sequence {a_n} where S_n is the sum of its first n terms,
    prove that if S_11 = 22π/3, then tan(a_6) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 11 = 22 * Real.pi / 3 →             -- Given condition
  Real.tan (a 6) = -Real.sqrt 3 := by
sorry

end arithmetic_sequence_tangent_l3336_333601


namespace initial_books_l3336_333638

theorem initial_books (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 77 → additional = 23 → total = initial + additional → initial = 54 := by
sorry

end initial_books_l3336_333638


namespace tangent_line_at_2_monotonicity_intervals_l3336_333615

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (m c : ℝ), ∀ x y, y = m * (x - 2) + f 2 ↔ 12 * x - y - 17 = 0 := by sorry

-- Theorem for intervals of monotonicity
theorem monotonicity_intervals :
  (∀ x, x < 0 → (f' x > 0)) ∧
  (∀ x, 0 < x ∧ x < 1 → (f' x < 0)) ∧
  (∀ x, x > 1 → (f' x > 0)) := by sorry

end tangent_line_at_2_monotonicity_intervals_l3336_333615


namespace quadratic_roots_ratio_l3336_333682

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_roots_ratio_l3336_333682


namespace triangle_base_length_l3336_333651

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 24 →
  height = 8 →
  area = (base * height) / 2 →
  base = 6 := by
sorry

end triangle_base_length_l3336_333651


namespace class_size_l3336_333696

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 24) :
  french + german - both + neither = 78 := by
  sorry

end class_size_l3336_333696


namespace tangent_line_minimum_value_l3336_333660

/-- The minimum value of 1/a^2 + 1/b^2 for a line tangent to a circle -/
theorem tangent_line_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + 2 * b * y + 2 = 0 ∧ x^2 + y^2 = 2) :
  (1 / a^2 + 1 / b^2 : ℝ) ≥ 9/2 ∧ 
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a₀ * x + 2 * b₀ * y + 2 = 0 ∧ x^2 + y^2 = 2) ∧
    1 / a₀^2 + 1 / b₀^2 = 9/2) :=
by sorry

end tangent_line_minimum_value_l3336_333660


namespace shoe_pairs_problem_l3336_333665

theorem shoe_pairs_problem (n : ℕ) (h : n > 0) :
  (1 : ℚ) / (2 * n - 1 : ℚ) = 1 / 5 → n = 3 := by
  sorry

end shoe_pairs_problem_l3336_333665


namespace sets_partition_integers_l3336_333657

theorem sets_partition_integers (A B : Set ℤ) 
  (h1 : A ∪ B = (Set.univ : Set ℤ))
  (h2 : ∀ x : ℤ, x ∈ A → x - 1 ∈ B)
  (h3 : ∀ x y : ℤ, x ∈ B → y ∈ B → x + y ∈ A) :
  A = {x : ℤ | ∃ k : ℤ, x = 2 * k} ∧ 
  B = {x : ℤ | ∃ k : ℤ, x = 2 * k + 1} :=
by sorry

end sets_partition_integers_l3336_333657


namespace stating_probability_all_types_proof_l3336_333678

/-- Represents the probability of finding all three types of dolls in 4 blind boxes -/
def probability_all_types (ratio_A ratio_B ratio_C : ℕ) : ℝ :=
  let total := ratio_A + ratio_B + ratio_C
  let p_A := ratio_A / total
  let p_B := ratio_B / total
  let p_C := ratio_C / total
  4 * p_C * 3 * p_B * p_A^2 + 4 * p_C * 3 * p_B^2 * p_A + 6 * p_C^2 * 2 * p_B * p_A

/-- 
Theorem stating that given the production ratio of dolls A:B:C as 6:3:1, 
the probability of finding all three types of dolls when buying 4 blind boxes at once is 0.216
-/
theorem probability_all_types_proof :
  probability_all_types 6 3 1 = 0.216 := by
  sorry

end stating_probability_all_types_proof_l3336_333678


namespace stack_surface_area_l3336_333653

/-- Calculates the external surface area of a stack of cubes -/
def external_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  let adjusted_areas := surface_areas.zip side_lengths
    |> List.map (fun (area, s) => area - s^2)
  adjusted_areas.sum + 6 * (volumes.head!^(1/3))^2

/-- The volumes of the cubes in the stack -/
def cube_volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

theorem stack_surface_area :
  external_surface_area cube_volumes = 1021 := by
  sorry

end stack_surface_area_l3336_333653


namespace calculate_y_l3336_333679

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.Z = 4 * t.X ∧ t.x = 35 ∧ t.z = 60

-- Define the Law of Sines
def law_of_sines (t : Triangle) : Prop :=
  t.x / Real.sin t.X = t.y / Real.sin t.Y ∧
  t.y / Real.sin t.Y = t.z / Real.sin t.Z ∧
  t.z / Real.sin t.Z = t.x / Real.sin t.X

-- Theorem statement
theorem calculate_y (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : law_of_sines t) : 
  ∃ y : Real, t.y = y :=
sorry

end calculate_y_l3336_333679


namespace square_difference_problem_l3336_333600

theorem square_difference_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) :
  |x^2 - y^2| = 108 := by
  sorry

end square_difference_problem_l3336_333600


namespace percent_of_a_is_3b_l3336_333661

theorem percent_of_a_is_3b (a b : ℝ) (h : a = 1.5 * b) : (3 * b) / a * 100 = 200 := by
  sorry

end percent_of_a_is_3b_l3336_333661


namespace sqrt_expression_equals_negative_two_l3336_333616

theorem sqrt_expression_equals_negative_two :
  Real.sqrt 24 + (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 + Real.sqrt 2)^2 = -2 := by
  sorry

end sqrt_expression_equals_negative_two_l3336_333616


namespace estimations_correct_l3336_333617

/-- A function that performs rounding to the nearest hundred. -/
def roundToHundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

/-- The acceptable error margin for estimation. -/
def ε : ℚ := 100

/-- Theorem stating that the estimations are correct within the error margin. -/
theorem estimations_correct :
  let e1 := |212 + 384 - roundToHundred 212 - roundToHundred 384|
  let e2 := |903 - 497 - (roundToHundred 903 - roundToHundred 497)|
  let e3 := |206 + 3060 - roundToHundred 206 - roundToHundred 3060|
  let e4 := |523 + 386 - roundToHundred 523 - roundToHundred 386|
  (e1 ≤ ε) ∧ (e2 ≤ ε) ∧ (e3 ≤ ε) ∧ (e4 ≤ ε) := by
  sorry

end estimations_correct_l3336_333617
