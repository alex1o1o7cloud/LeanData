import Mathlib

namespace dog_barks_theorem_l3870_387018

/-- The number of times a single dog barks per minute -/
def single_dog_barks_per_minute : ℕ := 30

/-- The number of dogs -/
def number_of_dogs : ℕ := 2

/-- The duration of barking in minutes -/
def duration : ℕ := 10

/-- The total number of barks from all dogs -/
def total_barks : ℕ := 600

theorem dog_barks_theorem :
  single_dog_barks_per_minute * number_of_dogs * duration = total_barks :=
by sorry

end dog_barks_theorem_l3870_387018


namespace sum_of_fractions_equals_three_halves_l3870_387005

/-- Given real numbers a, b, and c satisfying the conditions,
    prove that (a/(b+c)) + (b/(c+a)) + (c/(a+b)) = 3/2 -/
theorem sum_of_fractions_equals_three_halves
  (a b c : ℝ)
  (h1 : a^3 + b^3 + c^3 = 3*a*b*c)
  (h2 : Matrix.det !![a, b, c; c, a, b; b, c, a] = 0) :
  a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 :=
by sorry

end sum_of_fractions_equals_three_halves_l3870_387005


namespace common_ratio_is_two_l3870_387055

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem common_ratio_is_two 
  (a₁ : ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n : ℕ, geometric_sequence a₁ q n > 0)
  (h_product : geometric_sequence a₁ q 1 * geometric_sequence a₁ q 5 = 16)
  (h_first_term : a₁ = 2) :
  q = 2 := by
sorry

end common_ratio_is_two_l3870_387055


namespace negation_of_universal_proposition_l3870_387094

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_proposition_l3870_387094


namespace arithmetic_triangle_inradius_l3870_387049

/-- A triangle with sides in arithmetic progression and an inscribed circle -/
structure ArithmeticTriangle where
  -- The three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The sides form an arithmetic progression
  progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d
  -- The triangle is valid (sum of any two sides is greater than the third)
  valid : a + b > c ∧ b + c > a ∧ c + a > b
  -- The triangle has positive area
  positive_area : a > 0 ∧ b > 0 ∧ c > 0
  -- The inscribed circle exists
  inradius : ℝ
  -- One of the altitudes
  altitude : ℝ

/-- 
The radius of the inscribed circle of a triangle with sides in arithmetic progression 
is equal to 1/3 of one of its altitudes
-/
theorem arithmetic_triangle_inradius (t : ArithmeticTriangle) : 
  t.inradius = (1/3) * t.altitude := by sorry

end arithmetic_triangle_inradius_l3870_387049


namespace original_price_l3870_387046

-- Define the discount rate
def discount_rate : ℝ := 0.4

-- Define the discounted price
def discounted_price : ℝ := 120

-- Theorem stating the original price
theorem original_price : 
  ∃ (price : ℝ), price * (1 - discount_rate) = discounted_price ∧ price = 200 :=
by
  sorry

end original_price_l3870_387046


namespace abs_neg_one_eq_one_l3870_387084

theorem abs_neg_one_eq_one : |(-1 : ℚ)| = 1 := by
  sorry

end abs_neg_one_eq_one_l3870_387084


namespace taxi_charge_calculation_l3870_387025

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

theorem taxi_charge_calculation :
  let initial_fee : ℚ := 9/4  -- $2.25
  let charge_per_increment : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee charge_per_increment increment_distance trip_distance = 27/5  -- $5.40
:= by sorry

end taxi_charge_calculation_l3870_387025


namespace average_of_combined_results_l3870_387051

theorem average_of_combined_results (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg₂ : ℝ) :
  n₁ = 60 →
  n₂ = 40 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 48 := by
  sorry

end average_of_combined_results_l3870_387051


namespace B_elements_l3870_387064

def B : Set ℤ := {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} := by sorry

end B_elements_l3870_387064


namespace satellite_sensor_ratio_l3870_387090

theorem satellite_sensor_ratio (total_sensors : ℝ) (non_upgraded_per_unit : ℝ) : 
  total_sensors > 0 →
  non_upgraded_per_unit ≥ 0 →
  (24 * non_upgraded_per_unit + 0.25 * total_sensors = total_sensors) →
  (non_upgraded_per_unit / (0.25 * total_sensors) = 1 / 8) :=
by
  sorry

end satellite_sensor_ratio_l3870_387090


namespace frequency_calculation_l3870_387069

theorem frequency_calculation (sample_size : ℕ) (frequency_rate : ℚ) (h1 : sample_size = 1000) (h2 : frequency_rate = 0.4) :
  (sample_size : ℚ) * frequency_rate = 400 := by
  sorry

end frequency_calculation_l3870_387069


namespace divisibility_condition_l3870_387063

theorem divisibility_condition (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ (x + y^3) % (x^2 + y^2) = 0 →
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨
  (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -1) := by
sorry

end divisibility_condition_l3870_387063


namespace correct_factorization_l3870_387075

theorem correct_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end correct_factorization_l3870_387075


namespace no_integer_solutions_for_equation_l3870_387091

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end no_integer_solutions_for_equation_l3870_387091


namespace repeating_decimal_sum_l3870_387047

/-- Represents a repeating decimal of the form 0.abab̄ab -/
def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + a + b : ℚ) / 9999

/-- Represents a repeating decimal of the form 0.abcabc̄abc -/
def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 999999

/-- The main theorem stating that if the sum of the two repeating decimals
    equals 33/37, then abc must be 447 -/
theorem repeating_decimal_sum (a b c : ℕ) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10) 
  (h_sum : repeating_decimal_2 a b + repeating_decimal_3 a b c = 33/37) :
  100 * a + 10 * b + c = 447 := by
  sorry

end repeating_decimal_sum_l3870_387047


namespace middle_book_pages_l3870_387098

def longest_book : ℕ := 396

def shortest_book : ℕ := longest_book / 4

def middle_book : ℕ := 3 * shortest_book

theorem middle_book_pages : middle_book = 297 := by
  sorry

end middle_book_pages_l3870_387098


namespace max_value_of_expression_l3870_387097

/-- Given that a, b, and c are distinct elements from the set {1, 2, 4},
    the maximum value of (a / 2) / (b / c) is 8. -/
theorem max_value_of_expression (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (a / 2 : ℚ) / (b / c : ℚ) ≤ 8 ∧ 
  ∃ (x y z : ℕ), x ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 y ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 z ∈ ({1, 2, 4} : Set ℕ) ∧ 
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 (x / 2 : ℚ) / (y / z : ℚ) = 8 :=
sorry

end max_value_of_expression_l3870_387097


namespace distance_between_trees_l3870_387077

/-- Given a yard of length 150 meters with 11 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 150 →
  num_trees = 11 →
  let num_segments := num_trees - 1
  yard_length / num_segments = 15 := by
  sorry

end distance_between_trees_l3870_387077


namespace function_inequality_condition_l3870_387074

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 - 4*x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 1| < a) ↔
  b ≤ Real.sqrt a := by
  sorry

end function_inequality_condition_l3870_387074


namespace rebus_solution_l3870_387089

theorem rebus_solution :
  ∃! (A B G D V : ℕ),
    A * B + 8 = 3 * B ∧
    G * D + B = V ∧
    G * B + 3 = A * D ∧
    A = 2 ∧ B = 7 ∧ G = 1 ∧ D = 0 ∧ V = 15 := by
  sorry

end rebus_solution_l3870_387089


namespace lennon_reimbursement_l3870_387058

/-- Calculates the total reimbursement for a sales rep given daily mileage and reimbursement rate -/
def calculate_reimbursement (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) (rate : ℚ) : ℚ :=
  (monday + tuesday + wednesday + thursday + friday : ℚ) * rate

/-- Proves that the total reimbursement for Lennon's mileage is $36 -/
theorem lennon_reimbursement :
  calculate_reimbursement 18 26 20 20 16 (36/100) = 36 := by
  sorry

end lennon_reimbursement_l3870_387058


namespace largest_divisor_power_l3870_387003

theorem largest_divisor_power (k : ℕ+) : 
  (∀ m : ℕ+, m ≤ k → (1991 : ℤ)^(m : ℕ) ∣ 1990^19911992 + 1992^19911990) ∧ 
  ¬((1991 : ℤ)^((k + 1) : ℕ) ∣ 1990^19911992 + 1992^19911990) → 
  k = 1 := by
sorry

end largest_divisor_power_l3870_387003


namespace larger_number_problem_l3870_387026

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end larger_number_problem_l3870_387026


namespace ratio_of_sum_and_difference_l3870_387038

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end ratio_of_sum_and_difference_l3870_387038


namespace exp_sum_geq_sin_cos_square_l3870_387032

theorem exp_sum_geq_sin_cos_square (x : ℝ) : Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end exp_sum_geq_sin_cos_square_l3870_387032


namespace saltwater_volume_l3870_387016

/-- Proves that the initial volume of a saltwater solution is 160 gallons given specific conditions --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x = x * 0.20) ∧ 
  (0.20 * x + 16 = (1/3) * (3/4 * x + 8 + 16)) ∧ 
  (x = 160) := by
sorry

end saltwater_volume_l3870_387016


namespace extrema_relations_l3870_387088

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x^2 + 1)

theorem extrema_relations (a b : ℝ) 
  (h1 : ∀ x, f x ≥ a) 
  (h2 : ∃ x, f x = a)
  (h3 : ∀ x, f x ≤ b) 
  (h4 : ∃ x, f x = b) :
  (∀ x, (x^3 - 1) / (x^6 + 1) ≥ a) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = a) ∧
  (∀ x, (x^3 - 1) / (x^6 + 1) ≤ b) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≥ -b) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≤ -a) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -a) :=
by sorry

end extrema_relations_l3870_387088


namespace geometric_sequence_increasing_condition_l3870_387085

/-- A geometric sequence with first term a and common ratio r -/
def GeometricSequence (a r : ℝ) : ℕ → ℝ :=
  fun n => a * r^(n - 1)

/-- An increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → f n < f m

theorem geometric_sequence_increasing_condition (a r : ℝ) :
  (IsIncreasing (GeometricSequence a r) → 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5) ∧
  (∃ a r : ℝ, 
    GeometricSequence a r 1 < GeometricSequence a r 3 ∧ 
    GeometricSequence a r 3 < GeometricSequence a r 5 ∧
    ¬IsIncreasing (GeometricSequence a r)) :=
by sorry


end geometric_sequence_increasing_condition_l3870_387085


namespace partition_condition_l3870_387068

/-- A partition of ℕ* into n sets satisfying the given conditions -/
structure Partition (a : ℝ) where
  n : ℕ+
  sets : Fin n → Set ℕ+
  disjoint : ∀ i j, i ≠ j → Disjoint (sets i) (sets j)
  cover : (⋃ i, sets i) = Set.univ
  infinite : ∀ i, Set.Infinite (sets i)
  difference : ∀ i x y, x ∈ sets i → y ∈ sets i → x > y → x - y ≥ a ^ (i : ℕ)

/-- The main theorem -/
theorem partition_condition (a : ℝ) : 
  (∃ p : Partition a, True) → a < 2 := by
  sorry

end partition_condition_l3870_387068


namespace jack_afternoon_emails_l3870_387061

/-- The number of emails Jack received in different parts of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Given information about Jack's email count -/
def jack_emails : EmailCount where
  morning := 5
  afternoon := 13 - 5
  evening := 72

/-- Theorem stating that Jack received 8 emails in the afternoon -/
theorem jack_afternoon_emails :
  jack_emails.afternoon = 8 := by
  sorry

end jack_afternoon_emails_l3870_387061


namespace acceptable_quality_probability_l3870_387043

theorem acceptable_quality_probability (p1 p2 : ℝ) 
  (h1 : p1 = 0.01) 
  (h2 : p2 = 0.03) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = 0.960 := by
  sorry

end acceptable_quality_probability_l3870_387043


namespace rectangular_box_volume_l3870_387078

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 20)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
  sorry

end rectangular_box_volume_l3870_387078


namespace value_of_expression_l3870_387041

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end value_of_expression_l3870_387041


namespace inscribed_circle_radius_squared_l3870_387023

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of segment ER -/
  er : ℝ
  /-- Length of segment RF -/
  rf : ℝ
  /-- Length of segment GS -/
  gs : ℝ
  /-- Length of segment SH -/
  sh : ℝ
  /-- The circle is tangent to EF at R and to GH at S -/
  tangent_condition : True

/-- The theorem stating that the square of the radius of the inscribed circle is (3225/118)^2 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.er = 22)
  (h2 : c.rf = 21)
  (h3 : c.gs = 40)
  (h4 : c.sh = 35) :
  c.r^2 = (3225/118)^2 := by
  sorry

end inscribed_circle_radius_squared_l3870_387023


namespace total_highlighters_l3870_387081

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 9) (h2 : yellow = 8) (h3 : blue = 5) :
  pink + yellow + blue = 22 := by
  sorry

end total_highlighters_l3870_387081


namespace parallelogram_area_l3870_387030

/-- The area of a parallelogram with base length 3 and height 3 is 9 square units. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 3 → 
  height = 3 → 
  area = base * height → 
  area = 9 := by
sorry

end parallelogram_area_l3870_387030


namespace total_easter_eggs_l3870_387021

def clubHouseEggs : ℕ := 40
def parkEggs : ℕ := 25
def townHallEggs : ℕ := 15

theorem total_easter_eggs : 
  clubHouseEggs + parkEggs + townHallEggs = 80 := by
  sorry

end total_easter_eggs_l3870_387021


namespace x_intercept_of_line_l3870_387092

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 ∧ y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l3870_387092


namespace complement_A_intersect_B_l3870_387040

-- Define the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Finset ℕ := {1, 2, 3}

-- Define set B
def B : Finset ℕ := {2, 3, 4}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4} :=
by sorry

end complement_A_intersect_B_l3870_387040


namespace lawsuit_probability_difference_l3870_387029

def probability_win_lawsuit1 : ℝ := 0.3
def probability_win_lawsuit2 : ℝ := 0.5

theorem lawsuit_probability_difference :
  (1 - probability_win_lawsuit1) * (1 - probability_win_lawsuit2) - 
  (probability_win_lawsuit1 * probability_win_lawsuit2) = 0.2 := by
  sorry

end lawsuit_probability_difference_l3870_387029


namespace min_value_of_f_l3870_387083

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + 1/b) / (2*x^2 + 2*x + 1)

theorem min_value_of_f (b : ℝ) (h : b > 0) :
  ∃ c : ℝ, c = -4 ∧ ∀ x : ℝ, f b x ≥ c :=
by sorry

end min_value_of_f_l3870_387083


namespace inequality_proof_l3870_387013

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x*y / (x^2 + y^2 + 2*z^2) + y*z / (2*x^2 + y^2 + z^2) + z*x / (x^2 + 2*y^2 + z^2) ≤ 3/4 := by
  sorry

end inequality_proof_l3870_387013


namespace system_inequalities_solution_equation_solution_l3870_387076

-- Define the system of inequalities
def system_inequalities (x : ℝ) : Prop :=
  2 * (x - 1) ≥ -4 ∧ (3 * x - 6) / 2 < x - 1

-- Define the set of positive integer solutions
def positive_integer_solutions : Set ℕ :=
  {1, 2, 3}

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ 3 / (x - 2) = 5 / (2 - x) - 1

-- Theorem for the system of inequalities
theorem system_inequalities_solution :
  ∀ n : ℕ, n ∈ positive_integer_solutions ↔ system_inequalities (n : ℝ) :=
sorry

-- Theorem for the equation
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ x = -6 :=
sorry

end system_inequalities_solution_equation_solution_l3870_387076


namespace T_is_three_rays_l3870_387045

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 3 ∧ y - 2 ≤ 4) ∨
               (4 = y - 2 ∧ x + 3 ≤ 4) ∨
               (x + 3 = y - 2 ∧ 4 ≤ x + 3)}

-- Define the three rays with common endpoint (1,6)
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 ≤ 6}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 1 ∧ p.2 = 6}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 1 ∧ p.2 = p.1 + 5}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 :=
  sorry

end T_is_three_rays_l3870_387045


namespace problem_statement_l3870_387035

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end problem_statement_l3870_387035


namespace product_sequence_equals_32_l3870_387079

theorem product_sequence_equals_32 : 
  (1/4 : ℚ) * 8 * (1/16 : ℚ) * 32 * (1/64 : ℚ) * 128 * (1/256 : ℚ) * 512 * (1/1024 : ℚ) * 2048 = 32 := by
  sorry

end product_sequence_equals_32_l3870_387079


namespace probability_of_three_given_sum_fourteen_l3870_387015

-- Define a type for die outcomes
def DieOutcome := Fin 6

-- Define a type for a set of four tosses
def FourTosses := Fin 4 → DieOutcome

-- Function to calculate the sum of four tosses
def sumTosses (tosses : FourTosses) : Nat :=
  (tosses 0).val + (tosses 1).val + (tosses 2).val + (tosses 3).val + 4

-- Function to check if a set of tosses contains at least one 3
def hasThree (tosses : FourTosses) : Prop :=
  ∃ i, (tosses i).val = 2

-- Theorem statement
theorem probability_of_three_given_sum_fourteen (tosses : FourTosses) :
  sumTosses tosses = 14 → hasThree tosses := by
  sorry

#check probability_of_three_given_sum_fourteen

end probability_of_three_given_sum_fourteen_l3870_387015


namespace vector_collinear_same_direction_l3870_387002

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

/-- The theorem statement -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b ∧ same_direction a b → x = Real.sqrt 2 :=
by sorry

end vector_collinear_same_direction_l3870_387002


namespace larger_region_area_unit_circle_chord_l3870_387082

/-- The area of the larger region formed by a chord of length 1 on a unit circle -/
theorem larger_region_area_unit_circle_chord (chord_length : Real) (h : chord_length = 1) :
  let circle_area : Real := π
  let triangle_area : Real := (Real.sqrt 3) / 4
  let sector_area : Real := π / 6
  let segment_area : Real := sector_area - triangle_area
  let larger_region_area : Real := circle_area - segment_area
  larger_region_area = 5 * π / 6 + (Real.sqrt 3) / 4 := by
  sorry


end larger_region_area_unit_circle_chord_l3870_387082


namespace smallest_integer_y_smallest_integer_solution_l3870_387004

theorem smallest_integer_y (y : ℤ) : (7 - 5 * y < 22) ↔ (y > -3) :=
  sorry

theorem smallest_integer_solution : ∃ y : ℤ, (∀ z : ℤ, 7 - 5 * z < 22 → y ≤ z) ∧ (7 - 5 * y < 22) ∧ y = -2 :=
  sorry

end smallest_integer_y_smallest_integer_solution_l3870_387004


namespace g_range_l3870_387048

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 4 + 3 * Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 3 * Real.cos x ^ 2 - 9) /
  (Real.sin x - 1)

theorem g_range :
  ∀ x : ℝ, Real.sin x ≠ 1 → 2 ≤ g x ∧ g x < 15 := by
  sorry

end g_range_l3870_387048


namespace total_spending_correct_l3870_387006

-- Define the stores and their purchases
structure Store :=
  (items : List (String × Float))
  (discount : Float)
  (accessoryDeal : Option (Float × Float))
  (freeItem : Option Float)
  (shippingFee : Bool)

def stores : List Store := [
  ⟨[("shoes", 200)], 0.3, none, none, false⟩,
  ⟨[("shirts", 160), ("pants", 150)], 0.2, none, none, false⟩,
  ⟨[("jacket", 250), ("tie", 40), ("hat", 60)], 0, some (0.5, 0.5), none, false⟩,
  ⟨[("watch", 120), ("wallet", 49)], 0, none, some 49, true⟩,
  ⟨[("belt", 35), ("scarf", 45)], 0, none, none, true⟩
]

-- Define the overall discount and tax rates
def rewardsDiscount : Float := 0.05
def salesTax : Float := 0.08

-- Define the gift card amount
def giftCardAmount : Float := 50

-- Define the shipping fee
def shippingFee : Float := 5

-- Function to calculate the total spending
noncomputable def calculateTotalSpending (stores : List Store) (rewardsDiscount : Float) (salesTax : Float) (giftCardAmount : Float) (shippingFee : Float) : Float :=
  sorry

-- Theorem to prove
theorem total_spending_correct :
  calculateTotalSpending stores rewardsDiscount salesTax giftCardAmount shippingFee = 854.29 :=
sorry

end total_spending_correct_l3870_387006


namespace coloring_existence_l3870_387066

theorem coloring_existence : ∃ (f : ℕ → Bool), 
  ∀ (a : ℕ → ℕ) (d : ℕ),
    (∀ i : Fin 18, a i < a (i + 1)) →
    (∀ i j : Fin 18, a j - a i = d * (j - i)) →
    1 ≤ a 0 → a 17 ≤ 1986 →
    ∃ i j : Fin 18, f (a i) ≠ f (a j) := by
  sorry

end coloring_existence_l3870_387066


namespace no_arithmetic_mean_l3870_387072

theorem no_arithmetic_mean (f1 f2 f3 : ℚ) : 
  f1 = 5/8 ∧ f2 = 9/12 ∧ f3 = 7/10 →
  (f1 ≠ (f2 + f3) / 2) ∧ (f2 ≠ (f1 + f3) / 2) ∧ (f3 ≠ (f1 + f2) / 2) := by
  sorry

#check no_arithmetic_mean

end no_arithmetic_mean_l3870_387072


namespace real_part_of_complex_fraction_l3870_387086

theorem real_part_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z = -1 := by sorry

end real_part_of_complex_fraction_l3870_387086


namespace inscribed_circle_radius_l3870_387054

/-- Configuration of semicircles and inscribed circle -/
structure SemicircleConfig where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- The inscribed circle touches both semicircles and the diameter -/
def touches_all (c : SemicircleConfig) : Prop :=
  ∃ (O O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ),
    let (xₒ, yₒ) := O
    let (x₁, y₁) := O₁
    let (x₂, y₂) := O₂
    let (xₚ, yₚ) := P
    (xₒ - x₂)^2 + (yₒ - y₂)^2 = (c.R - c.x)^2 ∧  -- Larger semicircle touches inscribed circle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (c.r + c.x)^2 ∧  -- Smaller semicircle touches inscribed circle
    (x₂ - xₚ)^2 + (y₂ - yₚ)^2 = c.x^2           -- Inscribed circle touches diameter

/-- Main theorem: The radius of the inscribed circle is 8 cm -/
theorem inscribed_circle_radius
  (c : SemicircleConfig)
  (h₁ : c.R = 18)
  (h₂ : c.r = 9)
  (h₃ : touches_all c) :
  c.x = 8 :=
sorry

end inscribed_circle_radius_l3870_387054


namespace monotone_decreasing_implies_m_ge_one_l3870_387014

/-- A function f(x) = x^2 - 2mx + 1 that is monotonically decreasing on (-∞, 1) -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

/-- The property that f is monotonically decreasing on (-∞, 1) -/
def is_monotone_decreasing (m : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 1 → f m x > f m y

/-- The theorem stating that if f is monotonically decreasing on (-∞, 1), then m ≥ 1 -/
theorem monotone_decreasing_implies_m_ge_one (m : ℝ) 
  (h : is_monotone_decreasing m) : m ≥ 1 := by
  sorry

#check monotone_decreasing_implies_m_ge_one

end monotone_decreasing_implies_m_ge_one_l3870_387014


namespace largest_n_for_inequality_l3870_387050

theorem largest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (a b c d : ℝ), 
    (n + 2) * Real.sqrt (a^2 + b^2) + (n + 1) * Real.sqrt (a^2 + c^2) + (n + 1) * Real.sqrt (a^2 + d^2) ≥ n * (a + b + c + d)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (a b c d : ℝ), 
      (m + 2) * Real.sqrt (a^2 + b^2) + (m + 1) * Real.sqrt (a^2 + c^2) + (m + 1) * Real.sqrt (a^2 + d^2) < m * (a + b + c + d)) :=
by sorry

end largest_n_for_inequality_l3870_387050


namespace two_numbers_difference_l3870_387024

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24365)
  (b_div_5 : b % 5 = 0)
  (b_div_10_eq_2a : b / 10 = 2 * a) :
  b - a = 19931 :=
by sorry

end two_numbers_difference_l3870_387024


namespace largest_number_l3870_387011

theorem largest_number (a b c d e : ℝ) :
  a = (7 * 8)^(1/4)^(1/2) →
  b = (8 * 7^(1/3))^(1/4) →
  c = (7 * 8^(1/4))^(1/2) →
  d = (7 * 8^(1/4))^(1/3) →
  e = (8 * 7^(1/3))^(1/4) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end largest_number_l3870_387011


namespace minimum_area_of_reported_tile_l3870_387039

/-- Represents the reported dimension of a side of a tile -/
structure ReportedDimension where
  value : ℝ
  lower_bound : ℝ := value - 0.7
  upper_bound : ℝ := value + 0.7

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedTile where
  length : ReportedDimension
  width : ReportedDimension

def minimum_area (tile : ReportedTile) : ℝ :=
  tile.length.lower_bound * tile.width.lower_bound

theorem minimum_area_of_reported_tile (tile : ReportedTile) 
  (h1 : tile.length.value = 3) 
  (h2 : tile.width.value = 4) : 
  minimum_area tile = 7.59 := by
  sorry

#eval minimum_area { length := { value := 3 }, width := { value := 4 } }

end minimum_area_of_reported_tile_l3870_387039


namespace monotonic_increasing_interval_l3870_387059

/-- The function f(x) = (3 - x^2)e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotonic_increasing_interval (x : ℝ) : 
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end monotonic_increasing_interval_l3870_387059


namespace even_number_of_fours_l3870_387087

theorem even_number_of_fours (n₃ n₄ n₅ : ℕ) : 
  n₃ + n₄ + n₅ = 80 →
  3 * n₃ + 4 * n₄ + 5 * n₅ = 276 →
  Even n₄ := by
sorry

end even_number_of_fours_l3870_387087


namespace unique_number_ratio_l3870_387008

theorem unique_number_ratio : ∃! x : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
  sorry

end unique_number_ratio_l3870_387008


namespace project_monthly_allocations_l3870_387033

/-- Proves that the number of monthly allocations is 12 given the project budget conditions -/
theorem project_monthly_allocations
  (total_budget : ℕ)
  (months_passed : ℕ)
  (amount_spent : ℕ)
  (over_budget : ℕ)
  (h1 : total_budget = 12600)
  (h2 : months_passed = 6)
  (h3 : amount_spent = 6580)
  (h4 : over_budget = 280)
  (h5 : ∃ (monthly_allocation : ℕ), total_budget = monthly_allocation * (total_budget / monthly_allocation)) :
  total_budget / ((amount_spent - over_budget) / months_passed) = 12 := by
  sorry

end project_monthly_allocations_l3870_387033


namespace circle_center_and_radius_l3870_387099

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 9

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end circle_center_and_radius_l3870_387099


namespace pet_store_birds_l3870_387095

/-- The number of birds in a cage -/
def birds_in_cage (parrots parakeets finches cockatiels canaries lovebirds toucans : ℕ) : ℕ :=
  parrots + parakeets + finches + cockatiels + canaries + lovebirds + toucans

/-- The total number of birds in the pet store -/
def total_birds : ℕ :=
  birds_in_cage 6 2 0 0 0 0 0 +  -- Cage 1
  birds_in_cage 4 3 5 0 0 0 0 +  -- Cage 2
  birds_in_cage 2 4 0 1 0 0 0 +  -- Cage 3
  birds_in_cage 3 5 0 0 2 0 0 +  -- Cage 4
  birds_in_cage 7 0 0 0 0 4 0 +  -- Cage 5
  birds_in_cage 4 2 3 0 0 0 1    -- Cage 6

theorem pet_store_birds : total_birds = 58 := by
  sorry

end pet_store_birds_l3870_387095


namespace older_brother_age_l3870_387031

theorem older_brother_age (father_age : ℕ) (n : ℕ) (x : ℕ) : 
  father_age = 50 ∧ 
  2 * (x + n) = father_age + n ∧
  x + n ≤ father_age →
  x + n = 25 :=
by sorry

end older_brother_age_l3870_387031


namespace function_value_at_nine_l3870_387034

-- Define the function f(x) = k * x^(1/2)
def f (k : ℝ) (x : ℝ) : ℝ := k * (x ^ (1/2))

-- State the theorem
theorem function_value_at_nine (k : ℝ) : 
  f k 16 = 6 → f k 9 = 9/2 := by
  sorry

end function_value_at_nine_l3870_387034


namespace sum_of_factorials_last_two_digits_l3870_387010

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := (List.range 2012).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_of_factorials_last_two_digits :
  last_two_digits S = 13 := by sorry

end sum_of_factorials_last_two_digits_l3870_387010


namespace gardening_time_ratio_l3870_387022

/-- Proves that the ratio of time to plant one flower to time to mow one line is 1/4 --/
theorem gardening_time_ratio :
  ∀ (total_time mow_time plant_time : ℕ) 
    (lines flowers_per_row rows : ℕ) 
    (time_per_line : ℕ),
  total_time = 108 →
  lines = 40 →
  time_per_line = 2 →
  flowers_per_row = 7 →
  rows = 8 →
  mow_time = lines * time_per_line →
  plant_time = total_time - mow_time →
  (plant_time : ℚ) / (rows * flowers_per_row : ℚ) / (time_per_line : ℚ) = 1 / 4 := by
  sorry

end gardening_time_ratio_l3870_387022


namespace train_passing_platform_l3870_387028

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_passing_platform (train_length : ℝ) (time_cross_tree : ℝ) (platform_length : ℝ) :
  train_length = 1200 ∧ time_cross_tree = 120 ∧ platform_length = 1100 →
  (train_length + platform_length) / (train_length / time_cross_tree) = 230 := by
sorry

end train_passing_platform_l3870_387028


namespace water_depth_calculation_l3870_387044

def water_depth (dean_height ron_height : ℝ) : ℝ :=
  2 * dean_height

theorem water_depth_calculation (ron_height : ℝ) (h1 : ron_height = 14) :
  water_depth (ron_height - 8) ron_height = 12 := by
  sorry

end water_depth_calculation_l3870_387044


namespace cube_volume_from_surface_area_l3870_387042

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 294) :
  let s := Real.sqrt (S / 6)
  s ^ 3 = 343 := by
  sorry

end cube_volume_from_surface_area_l3870_387042


namespace max_value_of_expression_l3870_387071

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_value_of_expression (hf : ∀ x, f x ∈ Set.Icc (-3) 5) 
                                 (hg : ∀ x, g x ∈ Set.Icc (-4) 2) :
  ∃ d, d = 45 ∧ ∀ x, 2 * f x * g x + f x ≤ d ∧ 
  ∃ y, 2 * f y * g y + f y = d :=
sorry

end max_value_of_expression_l3870_387071


namespace trig_identity_l3870_387007

theorem trig_identity (α : ℝ) (h : Real.cos (75 * π / 180 + α) = 1/3) :
  Real.sin (60 * π / 180 + 2*α) = 7/9 := by
  sorry

end trig_identity_l3870_387007


namespace parallel_line_through_point_l3870_387001

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let given_line : Line := { a := 3, b := 1, c := -1 }
  let parallel_line : Line := { a := 3, b := 1, c := -5 }
  let point : (ℝ × ℝ) := (1, 2)
  parallel given_line parallel_line ∧
  point_on_line point.1 point.2 parallel_line :=
by sorry

end parallel_line_through_point_l3870_387001


namespace one_in_M_l3870_387020

def M : Set ℕ := {0, 1, 2}

theorem one_in_M : 1 ∈ M := by
  sorry

end one_in_M_l3870_387020


namespace min_reciprocal_sum_min_value_is_two_l3870_387080

theorem min_reciprocal_sum (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h1 : b > 0) (h2 : a + b = 2) :
  1/a + 1/b = 2 :=
by sorry

end min_reciprocal_sum_min_value_is_two_l3870_387080


namespace min_value_theorem_l3870_387036

theorem min_value_theorem (x y a : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y) ≥ 6*Real.sqrt 3) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → 3*x + a*y ≥ 6*Real.sqrt 3) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end min_value_theorem_l3870_387036


namespace die_faces_count_l3870_387057

-- Define the probability of all five dice showing the same number
def probability : ℝ := 0.0007716049382716049

-- Define the number of dice
def num_dice : ℕ := 5

-- Theorem: The number of faces on each die is 10
theorem die_faces_count : 
  ∃ (n : ℕ), n > 0 ∧ (1 : ℝ) / n ^ num_dice = probability ∧ n = 10 :=
sorry

end die_faces_count_l3870_387057


namespace strictly_increasing_quadratic_function_l3870_387060

theorem strictly_increasing_quadratic_function (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → (x^2 - a*x) < (y^2 - a*y)) ↔ a ≤ 2 :=
by sorry

end strictly_increasing_quadratic_function_l3870_387060


namespace marbles_given_to_mary_l3870_387019

def initial_marbles : ℕ := 64
def remaining_marbles : ℕ := 50

theorem marbles_given_to_mary :
  initial_marbles - remaining_marbles = 14 :=
by sorry

end marbles_given_to_mary_l3870_387019


namespace smallest_dual_base_representation_l3870_387070

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  n = 2 * a + 1 ∧
  n = b + 2

theorem smallest_dual_base_representation :
  (is_valid_representation 7) ∧
  (∀ m : ℕ, m < 7 → ¬(is_valid_representation m)) :=
by sorry

end smallest_dual_base_representation_l3870_387070


namespace vodka_mixture_profit_l3870_387056

/-- Profit percentage of a mixture of two vodkas -/
def mixture_profit_percentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  ((profit1 * increase1 + profit2 * increase2) / 2)

theorem vodka_mixture_profit :
  let initial_profit1 : ℚ := 40 / 100
  let initial_profit2 : ℚ := 20 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixture_profit_percentage initial_profit1 initial_profit2 increase1 increase2 = 13 / 30 := by
  sorry

#eval (13 / 30 : ℚ)

end vodka_mixture_profit_l3870_387056


namespace problem_solution_l3870_387093

theorem problem_solution (x y : ℝ) (h1 : y = Real.log (2 * x)) (h2 : x + y = 2) :
  (Real.exp x + Real.exp y > 2 * Real.exp 1) ∧ (x * Real.log x + y * Real.log y > 0) := by
  sorry

end problem_solution_l3870_387093


namespace three_card_selection_l3870_387012

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def ways_to_pick_three_cards : ℕ := standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2)

theorem three_card_selection :
  ways_to_pick_three_cards = 132600 :=
sorry

end three_card_selection_l3870_387012


namespace factors_180_multiples_15_l3870_387053

/-- A function that returns the number of positive integers that are both factors of n and multiples of m -/
def count_common_factors_multiples (n m : ℕ) : ℕ :=
  (Finset.filter (λ x => n % x = 0 ∧ x % m = 0) (Finset.range n)).card

/-- Theorem stating that the number of positive integers that are both factors of 180 and multiples of 15 is 6 -/
theorem factors_180_multiples_15 : count_common_factors_multiples 180 15 = 6 := by
  sorry

end factors_180_multiples_15_l3870_387053


namespace no_integer_solutions_l3870_387009

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^3 + 4*x^2 + x = 18*y^3 + 18*y^2 + 6*y + 3 := by
  sorry

end no_integer_solutions_l3870_387009


namespace not_pythagorean_triple_l3870_387052

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : ¬ is_pythagorean_triple 7 25 26 := by
  sorry

end not_pythagorean_triple_l3870_387052


namespace product_of_one_plus_roots_l3870_387065

theorem product_of_one_plus_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end product_of_one_plus_roots_l3870_387065


namespace arithmetic_calculation_l3870_387037

theorem arithmetic_calculation : (2^3 * 3 * 5) + (18 / 2) = 129 := by
  sorry

end arithmetic_calculation_l3870_387037


namespace inequality_system_solutions_l3870_387027

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ (x : ℤ), (x > m ∧ x < 8) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)) ↔
  (4 ≤ m ∧ m < 5) :=
by sorry

end inequality_system_solutions_l3870_387027


namespace complex_magnitude_l3870_387096

theorem complex_magnitude (z : ℂ) (h : (z + 2) / (z - 2) = Complex.I) : Complex.abs z = 2 := by
  sorry

end complex_magnitude_l3870_387096


namespace trajectory_of_Q_l3870_387062

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M on circle C
def point_M (x₀ y₀ : ℝ) : Prop := circle_C x₀ y₀

-- Define the vector ON
def vector_ON (y₀ : ℝ) : ℝ × ℝ := (0, y₀)

-- Define the vector OQ as the sum of OM and ON
def vector_OQ (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, 2 * y₀)

-- State the theorem
theorem trajectory_of_Q (x y : ℝ) :
  (∃ x₀ y₀ : ℝ, point_M x₀ y₀ ∧ vector_OQ x₀ y₀ = (x, y)) →
  x^2/4 + y^2/16 = 1 :=
sorry

end trajectory_of_Q_l3870_387062


namespace product_of_sum_and_sum_of_cubes_l3870_387000

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by
sorry

end product_of_sum_and_sum_of_cubes_l3870_387000


namespace star_vertex_angle_formula_l3870_387067

/-- The angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
def starVertexAngle (n : ℕ) : ℚ :=
  (n - 4) * 180 / n

/-- Theorem stating the angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
theorem star_vertex_angle_formula (n : ℕ) (h : n > 2) :
  starVertexAngle n = (n - 4) * 180 / n :=
by sorry

end star_vertex_angle_formula_l3870_387067


namespace intersection_of_M_and_N_l3870_387017

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end intersection_of_M_and_N_l3870_387017


namespace angle_range_l3870_387073

theorem angle_range (α : Real) 
  (h1 : α > 0 ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > 0) 
  (h3 : Real.cos α < 0) : 
  α > Real.pi / 2 ∧ α < Real.pi :=
by sorry

end angle_range_l3870_387073
