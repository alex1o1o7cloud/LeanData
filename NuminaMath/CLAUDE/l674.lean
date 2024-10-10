import Mathlib

namespace derivative_f_at_one_l674_67406

/-- The function f(x) -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

/-- Theorem stating that the derivative of f(x) at x = 1 is 3 -/
theorem derivative_f_at_one : 
  deriv f 1 = 3 := by sorry

end derivative_f_at_one_l674_67406


namespace minimum_bailing_rate_l674_67471

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (max_water_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 8)
  (h4 : max_water_capacity = 50)
  : ∃ (min_bailing_rate : ℝ),
    min_bailing_rate ≥ 7 ∧
    (distance_to_shore / rowing_speed) * (water_intake_rate - min_bailing_rate) ≤ max_water_capacity ∧
    ∀ (r : ℝ), r < min_bailing_rate →
      (distance_to_shore / rowing_speed) * (water_intake_rate - r) > max_water_capacity :=
by sorry

end minimum_bailing_rate_l674_67471


namespace cd_purchase_remaining_money_l674_67412

theorem cd_purchase_remaining_money (total_money : ℚ) (num_cds : ℕ) (cd_price : ℚ) :
  (total_money / 5 = num_cds / 3 * cd_price) →
  (total_money - num_cds * cd_price) / total_money = 2 / 5 := by
  sorry

end cd_purchase_remaining_money_l674_67412


namespace range_of_m_l674_67463

/-- Proposition p: x < -2 or x > 10 -/
def p (x : ℝ) : Prop := x < -2 ∨ x > 10

/-- Proposition q: 1-m ≤ x ≤ 1+m^2 -/
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

/-- ¬p is a sufficient but not necessary condition for q -/
def suff_not_nec (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → q x m) ∧ ∃ x, q x m ∧ p x

theorem range_of_m :
  {m : ℝ | suff_not_nec m} = {m : ℝ | m ≥ 3} :=
sorry

end range_of_m_l674_67463


namespace complement_of_A_l674_67481

-- Define the universal set U
def U : Finset ℕ := {1,2,3,4,5,6,7}

-- Define set A
def A : Finset ℕ := Finset.filter (fun x => 1 ≤ x ∧ x ≤ 6) U

-- Theorem statement
theorem complement_of_A : (U \ A) = {7} := by sorry

end complement_of_A_l674_67481


namespace arithmetic_sequence_sum_l674_67473

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 5 + a 6 + a 8 = 25 →
  a 2 + a 8 = 10 := by
  sorry

end arithmetic_sequence_sum_l674_67473


namespace problem_solution_l674_67492

theorem problem_solution (a : ℝ) (h : a^2 + a = 0) : a^2011 + a^2010 + 12 = 12 := by
  sorry

end problem_solution_l674_67492


namespace brothers_savings_l674_67472

def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def number_of_brothers : ℕ := 2
def isabelle_savings : ℕ := 5
def work_weeks : ℕ := 10
def weekly_earnings : ℕ := 3

def total_ticket_cost : ℕ := isabelle_ticket_cost + number_of_brothers * brother_ticket_cost

def isabelle_total_earnings : ℕ := isabelle_savings + work_weeks * weekly_earnings

theorem brothers_savings : 
  total_ticket_cost - isabelle_total_earnings = 5 := by sorry

end brothers_savings_l674_67472


namespace student_selection_l674_67467

/-- The number of ways to select 3 students from a group of 4 boys and 3 girls, 
    including both boys and girls. -/
theorem student_selection (boys : Nat) (girls : Nat) : 
  boys = 4 → girls = 3 → Nat.choose boys 2 * Nat.choose girls 1 + 
                         Nat.choose boys 1 * Nat.choose girls 2 = 30 := by
  sorry

#eval Nat.choose 4 2 * Nat.choose 3 1 + Nat.choose 4 1 * Nat.choose 3 2

end student_selection_l674_67467


namespace reflect_P_across_x_axis_l674_67490

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (-2, -3)

/-- Theorem: Reflecting the point P(-2,-3) across the x-axis results in the coordinates (-2, 3). -/
theorem reflect_P_across_x_axis :
  reflect_x P = (-2, 3) := by sorry

end reflect_P_across_x_axis_l674_67490


namespace episode_length_l674_67485

theorem episode_length
  (num_episodes : ℕ)
  (watching_hours_per_day : ℕ)
  (total_days : ℕ)
  (h1 : num_episodes = 90)
  (h2 : watching_hours_per_day = 2)
  (h3 : total_days = 15) :
  (total_days * watching_hours_per_day * 60) / num_episodes = 20 :=
by sorry

end episode_length_l674_67485


namespace complex_number_quadrant_l674_67400

theorem complex_number_quadrant (z : ℂ) : z = (2 + Complex.I) / 3 → z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l674_67400


namespace incorrect_calculation_l674_67465

theorem incorrect_calculation (a : ℝ) (n : ℕ) : 
  a^(2*n) * (a^(2*n))^3 / a^(4*n) ≠ a^2 :=
sorry

end incorrect_calculation_l674_67465


namespace surrounding_circles_radius_l674_67478

theorem surrounding_circles_radius (r : ℝ) : 
  (∃ (A B C D : ℝ × ℝ),
    -- Define the centers of the four surrounding circles
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2 ∧
    -- Ensure the surrounding circles touch the central circle
    (A.1^2 + A.2^2 = (r + 2)^2) ∧
    (B.1^2 + B.2^2 = (r + 2)^2) ∧
    (C.1^2 + C.2^2 = (r + 2)^2) ∧
    (D.1^2 + D.2^2 = (r + 2)^2) ∧
    -- Ensure the surrounding circles touch each other
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2*r)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*r)^2 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (2*r)^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end surrounding_circles_radius_l674_67478


namespace complex_power_110_deg_36_l674_67409

theorem complex_power_110_deg_36 :
  (Complex.exp (110 * π / 180 * Complex.I)) ^ 36 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end complex_power_110_deg_36_l674_67409


namespace distance_knoxville_to_los_angeles_l674_67449

/-- A point on the complex plane representing a city --/
structure City where
  coord : ℂ

/-- A map of cities on the complex plane that preserves distances --/
structure CityMap where
  los_angeles : City
  boston : City
  knoxville : City
  preserves_distances : True

theorem distance_knoxville_to_los_angeles (map : CityMap)
  (h1 : map.los_angeles.coord = 0)
  (h2 : map.boston.coord = 2600 * I)
  (h3 : map.knoxville.coord = 780 + 1040 * I) :
  Complex.abs (map.knoxville.coord - map.los_angeles.coord) = 1300 := by
  sorry

end distance_knoxville_to_los_angeles_l674_67449


namespace grid_block_selection_l674_67498

theorem grid_block_selection (n : ℕ) (k : ℕ) : 
  n = 7 → k = 4 → (n.choose k) * (n.choose k) * k.factorial = 29400 := by
  sorry

end grid_block_selection_l674_67498


namespace count_parallelograms_l674_67470

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram with vertices P, Q, R, S -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram using the shoelace formula -/
def area (p : Parallelogram) : ℚ :=
  (1 / 2 : ℚ) * |p.Q.x * p.S.y - p.S.x * p.Q.y|

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is on the line y = mx -/
def isOnLine (p : Point) (m : ℤ) : Prop :=
  p.y = m * p.x

/-- The main theorem to be proved -/
theorem count_parallelograms :
  let validParallelogram (p : Parallelogram) : Prop :=
    p.P = ⟨0, 0⟩ ∧
    isFirstQuadrant p.Q ∧
    isFirstQuadrant p.R ∧
    isFirstQuadrant p.S ∧
    isOnLine p.Q 2 ∧
    isOnLine p.S 3 ∧
    area p = 2000000
  (parallelograms : Finset Parallelogram) →
  (∀ p ∈ parallelograms, validParallelogram p) →
  parallelograms.card = 196 :=
sorry

end count_parallelograms_l674_67470


namespace f_negative_iff_x_in_unit_interval_l674_67442

/-- The function f(x) = x^2 - x^(1/2) is negative if and only if x is in the open interval (0, 1) -/
theorem f_negative_iff_x_in_unit_interval (x : ℝ) :
  x^2 - x^(1/2) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end f_negative_iff_x_in_unit_interval_l674_67442


namespace negation_of_tan_gt_sin_l674_67493

open Real

theorem negation_of_tan_gt_sin :
  (¬ (∀ x, -π/2 < x ∧ x < π/2 → tan x > sin x)) ↔
  (∃ x, -π/2 < x ∧ x < π/2 ∧ tan x ≤ sin x) := by sorry

end negation_of_tan_gt_sin_l674_67493


namespace complex_number_and_imaginary_root_l674_67458

theorem complex_number_and_imaginary_root (z : ℂ) (m : ℂ) : 
  (∃ (r : ℝ), z + Complex.I = r) →
  (∃ (s : ℝ), z / (1 - Complex.I) = s) →
  (∃ (t : ℝ), m = Complex.I * t) →
  (∃ (x : ℝ), (x^2 : ℂ) + x * (1 + z) - (3 * m - 1) * Complex.I = 0) →
  z = 1 - Complex.I ∧ m = -Complex.I := by
sorry

end complex_number_and_imaginary_root_l674_67458


namespace tan_three_pi_halves_minus_alpha_l674_67430

theorem tan_three_pi_halves_minus_alpha (α : Real) 
  (h : Real.cos (Real.pi - α) = -3/5) : 
  Real.tan (3/2 * Real.pi - α) = 3/4 ∨ Real.tan (3/2 * Real.pi - α) = -3/4 := by
  sorry

end tan_three_pi_halves_minus_alpha_l674_67430


namespace no_nontrivial_solutions_l674_67428

theorem no_nontrivial_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (h2p1 : Nat.Prime (2 * p + 1)) :
  ∀ x y z : ℤ, x^p + 2*y^p + 5*z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end no_nontrivial_solutions_l674_67428


namespace inequality_problems_l674_67433

theorem inequality_problems (x : ℝ) :
  ((-x^2 + 4*x - 4 < 0) ↔ (x ≠ 2)) ∧
  ((((1 - x) / (x - 5)) > 0) ↔ (1 < x ∧ x < 5)) := by
  sorry

end inequality_problems_l674_67433


namespace common_factor_proof_l674_67459

variables (a b c : ℕ+)

theorem common_factor_proof : Nat.gcd (4 * a^2 * b^2 * c) (6 * a * b^3) = 2 * a * b^2 := by
  sorry

end common_factor_proof_l674_67459


namespace divisibility_of_squares_sum_l674_67489

theorem divisibility_of_squares_sum (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 :=
by sorry

end divisibility_of_squares_sum_l674_67489


namespace jerry_firecrackers_l674_67446

theorem jerry_firecrackers (F : ℕ) : 
  (F ≥ 12) →
  (5 * (F - 12) / 6 = 30) →
  F = 48 := by sorry

end jerry_firecrackers_l674_67446


namespace plate_on_square_table_l674_67425

/-- Given a square table with a round plate, if the distances from the plate to the table edges
    on one side are 10 cm and 63 cm, and on the opposite side are 20 cm and x cm,
    then x = 53 cm. -/
theorem plate_on_square_table (x : ℝ) : x = 53 := by
  sorry

end plate_on_square_table_l674_67425


namespace range_of_a_range_of_b_l674_67455

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (b : ℝ) (x : ℝ) : ℝ := b*x + 5 - 2*b

-- Theorem for part 1
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  -8 ≤ a ∧ a ≤ 0 :=
sorry

-- Theorem for part 2
theorem range_of_b (b : ℝ) :
  (∀ x₁ ∈ Set.Icc (1 : ℝ) 4, ∃ x₂ ∈ Set.Icc (1 : ℝ) 4, g b x₁ = f 3 x₂) →
  -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end range_of_a_range_of_b_l674_67455


namespace range_of_a_l674_67469

-- Define the function f
def f (x : ℝ) : ℝ := x * (abs x + 4)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a^2) + f a < 0) → (-1 < a ∧ a < 0) :=
by sorry

end range_of_a_l674_67469


namespace rope_jumps_percentage_l674_67480

def rope_jumps : List ℕ := [50, 77, 83, 91, 93, 101, 87, 102, 111, 63, 117, 89, 121, 130, 133, 146, 88, 158, 177, 188]

def total_students : ℕ := 20

def in_range (x : ℕ) : Bool := 80 ≤ x ∧ x ≤ 100

def count_in_range (l : List ℕ) : ℕ := (l.filter in_range).length

theorem rope_jumps_percentage :
  (count_in_range rope_jumps : ℚ) / total_students * 100 = 30 := by
  sorry

end rope_jumps_percentage_l674_67480


namespace simplify_expression_l674_67429

theorem simplify_expression : (27 ^ (1/6) - Real.sqrt (6 + 3/4)) ^ 2 = 3/4 := by
  sorry

end simplify_expression_l674_67429


namespace distance_product_on_curve_l674_67408

/-- The curve C defined by xy = 2 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 2}

/-- The theorem stating that the product of distances from any point on C to the axes is 2 -/
theorem distance_product_on_curve (p : ℝ × ℝ) (h : p ∈ C) :
  |p.1| * |p.2| = 2 := by
  sorry


end distance_product_on_curve_l674_67408


namespace sqrt_54_minus_4_bounds_l674_67447

theorem sqrt_54_minus_4_bounds : 3 < Real.sqrt 54 - 4 ∧ Real.sqrt 54 - 4 < 4 := by
  sorry

end sqrt_54_minus_4_bounds_l674_67447


namespace binomial_parameters_determination_l674_67402

/-- A random variable X following a Binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a Binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a Binomial distribution X with EX = 8 and DX = 1.6, n = 100 and p = 0.08 -/
theorem binomial_parameters_determination :
  ∀ X : BinomialDistribution, 
    expectedValue X = 8 → 
    variance X = 1.6 → 
    X.n = 100 ∧ X.p = 0.08 := by
  sorry

end binomial_parameters_determination_l674_67402


namespace circles_intersect_l674_67427

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (1, 0)
def center_O2 : ℝ × ℝ := (0, 3)
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  let d := Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)
  (radius_O2 - radius_O1 < d) ∧ (d < radius_O1 + radius_O2) := by
  sorry


end circles_intersect_l674_67427


namespace passing_percentage_l674_67482

def total_marks : ℕ := 400
def student_marks : ℕ := 150
def failed_by : ℕ := 30

theorem passing_percentage : 
  (((student_marks + failed_by : ℚ) / total_marks) * 100 = 45) := by sorry

end passing_percentage_l674_67482


namespace mathematics_collections_l674_67413

def word : String := "MATHEMATICS"

def num_vowels : Nat := 4
def num_consonants : Nat := 7
def num_ts : Nat := 2

def vowels_fall_off : Nat := 3
def consonants_fall_off : Nat := 4

def distinct_collections : Nat := 220

theorem mathematics_collections :
  (word.length = num_vowels + num_consonants) →
  (num_vowels = 4) →
  (num_consonants = 7) →
  (num_ts = 2) →
  (vowels_fall_off = 3) →
  (consonants_fall_off = 4) →
  distinct_collections = 220 := by
  sorry

end mathematics_collections_l674_67413


namespace drill_bits_purchase_cost_l674_67426

/-- The total cost of a purchase with tax -/
def total_cost (num_sets : ℕ) (price_per_set : ℚ) (tax_rate : ℚ) : ℚ :=
  let pre_tax_cost := num_sets * price_per_set
  let tax := pre_tax_cost * tax_rate
  pre_tax_cost + tax

/-- Theorem: The total cost for 5 sets of drill bits at $6 each with 10% tax is $33 -/
theorem drill_bits_purchase_cost :
  total_cost 5 6 (1/10) = 33 := by
  sorry

end drill_bits_purchase_cost_l674_67426


namespace brandy_energy_drinks_l674_67419

/-- The number of energy drinks Brandy drank -/
def num_drinks : ℕ := 4

/-- The maximum safe amount of caffeine per day in mg -/
def max_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink in mg -/
def caffeine_per_drink : ℕ := 120

/-- The amount of additional caffeine Brandy can safely consume after drinking the energy drinks in mg -/
def remaining_caffeine : ℕ := 20

theorem brandy_energy_drinks :
  num_drinks * caffeine_per_drink + remaining_caffeine = max_caffeine :=
sorry

end brandy_energy_drinks_l674_67419


namespace max_product_sum_180_l674_67432

theorem max_product_sum_180 : 
  ∀ a b : ℤ, a + b = 180 → a * b ≤ 8100 := by
  sorry

end max_product_sum_180_l674_67432


namespace carrots_taken_l674_67443

theorem carrots_taken (initial_carrots remaining_carrots : ℕ) :
  initial_carrots = 6 →
  remaining_carrots = 3 →
  initial_carrots - remaining_carrots = 3 :=
by sorry

end carrots_taken_l674_67443


namespace fruit_boxes_problem_l674_67410

theorem fruit_boxes_problem (total_pears : ℕ) : 
  (∃ (fruits_per_box : ℕ), 
    fruits_per_box = 12 + total_pears / 9 ∧ 
    fruits_per_box = (12 + total_pears) / 3 ∧
    fruits_per_box = 16) := by
  sorry

end fruit_boxes_problem_l674_67410


namespace class_test_probability_l674_67476

theorem class_test_probability (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.25)
  (h3 : p_both = 0.20) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end class_test_probability_l674_67476


namespace disjunction_true_l674_67444

open Real

theorem disjunction_true : 
  (¬(∀ α : ℝ, sin (π - α) ≠ -sin α)) ∨ (∃ x : ℝ, x ≥ 0 ∧ sin x > x) := by
  sorry

end disjunction_true_l674_67444


namespace circle_chords_and_triangles_l674_67448

/-- Given 10 points on the circumference of a circle, prove the number of chords and triangles -/
theorem circle_chords_and_triangles (n : ℕ) (hn : n = 10) :
  (Nat.choose n 2 = 45) ∧ (Nat.choose n 3 = 120) := by
  sorry

#check circle_chords_and_triangles

end circle_chords_and_triangles_l674_67448


namespace sum_diff_difference_is_six_l674_67464

/-- A two-digit number with specific properties -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_two_digit : 10 ≤ 10 * tens + ones ∧ 10 * tens + ones < 100
  digit_ratio : ones = 2 * tens
  interchange_diff : 10 * ones + tens - (10 * tens + ones) = 36

/-- The difference between the sum and difference of digits for a TwoDigitNumber -/
def sum_diff_difference (n : TwoDigitNumber) : Nat :=
  (n.tens + n.ones) - (n.ones - n.tens)

theorem sum_diff_difference_is_six (n : TwoDigitNumber) :
  sum_diff_difference n = 6 := by
  sorry

end sum_diff_difference_is_six_l674_67464


namespace star_one_one_eq_neg_eleven_l674_67418

/-- A custom binary operation on real numbers -/
noncomputable def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y

/-- Theorem stating that given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : star a b 3 5 = 15) 
  (h2 : star a b 4 7 = 28) : 
  star a b 1 1 = -11 := by
  sorry

#check star_one_one_eq_neg_eleven

end star_one_one_eq_neg_eleven_l674_67418


namespace star_calculation_l674_67434

def star (a b : ℚ) : ℚ := (a + b) / 4

theorem star_calculation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_calculation_l674_67434


namespace perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l674_67415

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem 1: A line is perpendicular to a plane iff it's perpendicular to every line in the plane
theorem perpendicular_to_plane_iff_perpendicular_to_all_lines 
  (l : Line) (p : Plane) :
  perpendicular_to_plane l p ↔ 
  ∀ (m : Line), in_plane m p → perpendicular l m :=
sorry

-- Theorem 2: If a is parallel to b, and l is perpendicular to a, then l is perpendicular to b
theorem perpendicular_parallel_transitive 
  (a b l : Line) :
  parallel a b → perpendicular l a → perpendicular l b :=
sorry

end perpendicular_to_plane_iff_perpendicular_to_all_lines_perpendicular_parallel_transitive_l674_67415


namespace quadratic_equation_transformation_l674_67437

theorem quadratic_equation_transformation (x : ℝ) : 
  ((x - 1) * (x + 1) = 1) ↔ (x^2 - 2 = 0) := by
sorry

end quadratic_equation_transformation_l674_67437


namespace sector_central_angle_l674_67468

/-- The central angle of a sector with radius R and circumference 3R is 1 radian. -/
theorem sector_central_angle (R : ℝ) (R_pos : R > 0) : 
  let circumference := 3 * R
  let central_angle := circumference / R - 2
  central_angle = 1 := by sorry

end sector_central_angle_l674_67468


namespace sum_of_numbers_l674_67484

theorem sum_of_numbers : let numbers := [0.8, 1/2, 0.5]
  (∀ x ∈ numbers, x ≤ 2) →
  numbers.sum = 1.8 := by
  sorry

end sum_of_numbers_l674_67484


namespace overlapping_triangle_area_l674_67461

/-- Given a rectangle with length 8 and width 4, when folded along its diagonal,
    the area of the overlapping triangle is 10. -/
theorem overlapping_triangle_area (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  let diagonal := Real.sqrt (length ^ 2 + width ^ 2)
  let overlap_base := (length ^ 2 + width ^ 2) / (2 * length)
  let overlap_height := width
  let overlap_area := (1 / 2) * overlap_base * overlap_height
  overlap_area = 10 := by
sorry

end overlapping_triangle_area_l674_67461


namespace partial_fraction_sum_l674_67456

theorem partial_fraction_sum (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) →
  A + B + C + D + E = 0 := by
sorry

end partial_fraction_sum_l674_67456


namespace last_two_digits_sum_l674_67487

theorem last_two_digits_sum (n : ℕ) : (7^30 + 13^30) % 100 = 0 := by
  sorry

end last_two_digits_sum_l674_67487


namespace movie_theater_deal_l674_67477

/-- Movie theater deal problem -/
theorem movie_theater_deal (deal_price : ℝ) (ticket_price : ℝ) (savings : ℝ)
  (h1 : deal_price = 20)
  (h2 : ticket_price = 8)
  (h3 : savings = 2) :
  let popcorn_price := ticket_price - 3
  let total_normal_price := deal_price + savings
  let drink_price := (total_normal_price - ticket_price - popcorn_price) * (2/3)
  drink_price - popcorn_price = 1 := by sorry

end movie_theater_deal_l674_67477


namespace zoo_count_l674_67466

theorem zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 710 → 
  ∃ (birds mammals unique : ℕ), 
    birds + mammals + unique = total_heads ∧
    2 * birds + 4 * mammals + 3 * unique = total_legs ∧
    birds = 230 := by
  sorry

end zoo_count_l674_67466


namespace mint_code_is_6785_l674_67474

-- Define a function that maps characters to digits based on their position in GREAT MIND
def code_to_digit (c : Char) : Nat :=
  match c with
  | 'G' => 1
  | 'R' => 2
  | 'E' => 3
  | 'A' => 4
  | 'T' => 5
  | 'M' => 6
  | 'I' => 7
  | 'N' => 8
  | 'D' => 9
  | _ => 0

-- Define a function that converts a string to a number using the code
def code_to_number (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + code_to_digit c) 0

-- Theorem stating that MINT represents 6785
theorem mint_code_is_6785 : code_to_number "MINT" = 6785 := by
  sorry

end mint_code_is_6785_l674_67474


namespace subset_probability_l674_67483

def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def T : Finset Char := {'a', 'b', 'c'}

theorem subset_probability : 
  (Finset.filter (fun X => X ⊆ T) (Finset.powerset S)).card / (Finset.powerset S).card = 1 / 4 := by
  sorry

end subset_probability_l674_67483


namespace rational_sum_power_l674_67404

theorem rational_sum_power (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : 
  (n + m)^2023 = -1 := by
  sorry

end rational_sum_power_l674_67404


namespace correct_average_after_error_correction_l674_67407

theorem correct_average_after_error_correction (n : ℕ) (initial_avg : ℚ) (wrong_value correct_value : ℚ) :
  n = 10 →
  initial_avg = 23 →
  wrong_value = 26 →
  correct_value = 36 →
  (n : ℚ) * initial_avg + (correct_value - wrong_value) = n * 24 :=
by sorry

end correct_average_after_error_correction_l674_67407


namespace parallel_lines_sum_l674_67440

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  parallel : 1 / (-2) = 2 / n
  distance : 2 * Real.sqrt 5 = |m + 3| / Real.sqrt 5

/-- The sum of coefficients m and n for parallel lines with given properties -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = 3 := by
  sorry

end parallel_lines_sum_l674_67440


namespace knights_and_knaves_l674_67479

-- Define the type for individuals
inductive Person : Type
| A
| B
| C

-- Define the type for knight/knave status
inductive Status : Type
| Knight
| Knave

-- Function to determine if a person is a knight
def isKnight (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knight

-- Function to determine if a person is a knave
def isKnave (p : Person) (s : Person → Status) : Prop :=
  s p = Status.Knave

-- A's statement
def A_statement (s : Person → Status) : Prop :=
  isKnight Person.C s → isKnave Person.B s

-- C's statement
def C_statement (s : Person → Status) : Prop :=
  (isKnight Person.A s ∧ isKnave Person.C s) ∨ (isKnave Person.A s ∧ isKnight Person.C s)

-- Main theorem
theorem knights_and_knaves :
  ∃ (s : Person → Status),
    (∀ p, (isKnight p s → A_statement s = true) ∧ (isKnave p s → A_statement s = false)) ∧
    (∀ p, (isKnight p s → C_statement s = true) ∧ (isKnave p s → C_statement s = false)) ∧
    isKnave Person.A s ∧ isKnight Person.B s ∧ isKnight Person.C s :=
sorry

end knights_and_knaves_l674_67479


namespace percent_less_problem_l674_67435

theorem percent_less_problem (w x y z : ℝ) : 
  x = y * (1 - z / 100) →
  y = 1.4 * w →
  x = 5 * w / 4 →
  z = 10.71 := by
sorry

end percent_less_problem_l674_67435


namespace max_prob_at_one_l674_67424

def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_prob_at_one :
  let n : ℕ := 5
  let p : ℝ := 1/4
  ∀ k : ℕ, k ≠ 1 → k ≤ n → binomial_prob n 1 p > binomial_prob n k p :=
by sorry

end max_prob_at_one_l674_67424


namespace caravan_keepers_l674_67491

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ := by sorry

theorem caravan_keepers :
  let hens : ℕ := 50
  let goats : ℕ := 45
  let camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_head : ℕ := 1
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := hens * hen_feet + goats * goat_feet + camels * camel_feet
  let total_animal_heads : ℕ := hens + goats + camels
  let extra_feet : ℕ := 224
  num_keepers * keeper_feet + total_animal_feet = num_keepers * keeper_head + total_animal_heads + extra_feet →
  num_keepers = 15 := by sorry

end caravan_keepers_l674_67491


namespace downstream_distance_is_24km_l674_67475

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (scenario : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 24 km -/
theorem downstream_distance_is_24km 
  (scenario : SwimmingScenario)
  (h1 : scenario.upstream_distance = 12)
  (h2 : scenario.upstream_time = 6)
  (h3 : scenario.downstream_time = 6)
  (h4 : scenario.still_water_speed = 3) :
  downstream_distance scenario = 24 :=
sorry

end downstream_distance_is_24km_l674_67475


namespace smallest_integer_with_eight_factors_l674_67431

theorem smallest_integer_with_eight_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (factors : Finset ℕ), factors.card = 8 ∧ 
    (∀ m ∈ factors, m > 0 ∧ n % m = 0) ∧
    (∀ m : ℕ, m > 0 → n % m = 0 → m ∈ factors)) ∧
  (∀ k : ℕ, k > 0 → k < n →
    ¬(∃ (factors : Finset ℕ), factors.card = 8 ∧ 
      (∀ m ∈ factors, m > 0 ∧ k % m = 0) ∧
      (∀ m : ℕ, m > 0 → k % m = 0 → m ∈ factors))) :=
by sorry

end smallest_integer_with_eight_factors_l674_67431


namespace arithmetic_progression_relationship_l674_67496

theorem arithmetic_progression_relationship (x y z d : ℝ) : 
  (x + (y - z) ≠ y + (z - x) ∧ 
   y + (z - x) ≠ z + (x - y) ∧ 
   x + (y - z) ≠ z + (x - y)) →
  (x + (y - z) ≠ 0 ∧ y + (z - x) ≠ 0 ∧ z + (x - y) ≠ 0) →
  (y + (z - x)) - (x + (y - z)) = d →
  (z + (x - y)) - (y + (z - x)) = d →
  (x = y + d / 2 ∧ z = y + d) :=
by sorry

end arithmetic_progression_relationship_l674_67496


namespace sqrt_sum_comparison_l674_67439

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 5 > Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end sqrt_sum_comparison_l674_67439


namespace philip_intersections_l674_67441

theorem philip_intersections (crosswalks_per_intersection : ℕ) 
                              (lines_per_crosswalk : ℕ) 
                              (total_lines : ℕ) :
  crosswalks_per_intersection = 4 →
  lines_per_crosswalk = 20 →
  total_lines = 400 →
  total_lines / (crosswalks_per_intersection * lines_per_crosswalk) = 5 :=
by
  sorry

end philip_intersections_l674_67441


namespace class_ratio_and_total_l674_67414

theorem class_ratio_and_total (num_girls : ℕ) (num_boys : ℕ) : 
  (3 : ℚ) / 7 * num_boys = (6 : ℚ) / 11 * num_girls → 
  num_girls = 22 →
  (num_boys : ℚ) / num_girls = 14 / 11 ∧ num_boys + num_girls = 50 := by
  sorry

end class_ratio_and_total_l674_67414


namespace manuscript_cost_calculation_l674_67454

/-- Calculates the total cost of typing a manuscript with given conditions -/
def manuscript_typing_cost (total_pages : ℕ) (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * first_typing_rate
  let first_revision_cost := pages_revised_once * revision_rate
  let second_revision_cost := pages_revised_twice * revision_rate * 2
  initial_typing_cost + first_revision_cost + second_revision_cost

theorem manuscript_cost_calculation :
  manuscript_typing_cost 100 10 5 30 20 = 1350 := by
  sorry

end manuscript_cost_calculation_l674_67454


namespace election_percentage_l674_67417

theorem election_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 →
  winning_margin = 1504 →
  winning_percentage = 60 →
  (winning_percentage / 100) * total_votes - (total_votes - (winning_percentage / 100) * total_votes) = winning_margin :=
by sorry

end election_percentage_l674_67417


namespace polygon_problem_l674_67453

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- The sum of interior angles of a polygon -/
def interiorAngleSum (p : Polygon) : ℕ := 180 * (p.sides - 2)

/-- The number of diagonals in a polygon -/
def diagonalCount (p : Polygon) : ℕ := p.sides * (p.sides - 3) / 2

theorem polygon_problem (x y : Polygon) 
  (h1 : interiorAngleSum x + interiorAngleSum y = 1440)
  (h2 : x.sides * 3 = y.sides) :
  720 = 360 + 360 ∧ 
  x.sides = 3 ∧ 
  y.sides = 9 ∧ 
  diagonalCount y = 27 := by
  sorry

end polygon_problem_l674_67453


namespace complex_determinant_equation_l674_67445

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_determinant_equation :
  ∀ z : ℂ, determinant z i 1 i = 1 + i → z = 2 - i := by sorry

end complex_determinant_equation_l674_67445


namespace smallest_multiple_l674_67403

theorem smallest_multiple (n : ℕ) : n = 663 ↔ 
  n > 0 ∧ 
  n % 17 = 0 ∧ 
  (n - 6) % 73 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 17 = 0 → (m - 6) % 73 = 0 → m ≥ n :=
by sorry

end smallest_multiple_l674_67403


namespace students_on_korabelnaya_street_l674_67451

theorem students_on_korabelnaya_street (n : ℕ) : 
  n < 50 → 
  n % 7 = 0 → 
  n % 3 = 0 → 
  n % 2 = 0 → 
  n / 7 + n / 3 + n / 2 < n → 
  n - (n / 7 + n / 3 + n / 2) = 1 := by
sorry

end students_on_korabelnaya_street_l674_67451


namespace exists_binary_sequence_with_geometric_partial_sums_l674_67422

/-- A sequence where each term is either 0 or 1 -/
def BinarySequence := ℕ → Fin 2

/-- The partial sum of the first n terms of a BinarySequence -/
def PartialSum (a : BinarySequence) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => a i)

/-- A sequence of partial sums forms a geometric sequence -/
def IsGeometricSequence (S : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), ∀ n : ℕ, S (n + 1) = (r : ℚ) * S n

/-- There exists a BinarySequence whose partial sums form a geometric sequence -/
theorem exists_binary_sequence_with_geometric_partial_sums :
  ∃ (a : BinarySequence), IsGeometricSequence (PartialSum a) := by
  sorry

end exists_binary_sequence_with_geometric_partial_sums_l674_67422


namespace log_sum_problem_l674_67462

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_problem (x y z : ℝ) 
  (hx : log 3 (log 4 (log 5 x)) = 0)
  (hy : log 4 (log 5 (log 3 y)) = 0)
  (hz : log 5 (log 3 (log 4 z)) = 0) :
  x + y + z = 932 := by
  sorry

end log_sum_problem_l674_67462


namespace quadratic_root_problem_l674_67452

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) →
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1) :=
by sorry

end quadratic_root_problem_l674_67452


namespace correct_ages_l674_67420

/-- Represents the ages of Albert, Mary, Betty, and Carol -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ
  carol : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 10 ∧
  ages.carol = ages.betty + 3 ∧
  ages.carol = ages.mary / 2

/-- The theorem to prove -/
theorem correct_ages :
  ∃ (ages : Ages), satisfiesConditions ages ∧
    ages.albert = 20 ∧
    ages.mary = 10 ∧
    ages.betty = 2 ∧
    ages.carol = 5 := by
  sorry

end correct_ages_l674_67420


namespace problem_solution_l674_67488

theorem problem_solution : 
  let A : ℤ := -5 * -3
  let B : ℤ := 2 - 2
  A + B = 15 := by sorry

end problem_solution_l674_67488


namespace greatest_abcba_divisible_by_13_l674_67494

def is_valid_abcba (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n % 13 = 0 → n ≤ 83638 :=
by sorry

end greatest_abcba_divisible_by_13_l674_67494


namespace max_value_theorem_l674_67405

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
by sorry

end max_value_theorem_l674_67405


namespace okeydokey_receives_25_earthworms_l674_67423

/-- The number of apples Okeydokey paid -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey paid -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- Calculate the number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ :=
  (okeydokey_apples * total_earthworms) / (okeydokey_apples + artichokey_apples)

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_receives_25_earthworms :
  okeydokey_earthworms = 25 := by
  sorry

end okeydokey_receives_25_earthworms_l674_67423


namespace largest_product_of_three_l674_67495

def S : Finset Int := {-4, -3, -1, 3, 5}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    x * y * z ≤ 60) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 60) :=
by
  sorry

end largest_product_of_three_l674_67495


namespace equal_positive_integers_l674_67460

theorem equal_positive_integers (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ k : ℕ, b^n + n = k * (a^n + n)) : a = b := by
  sorry

end equal_positive_integers_l674_67460


namespace interest_rate_first_part_l674_67421

/-- Given a total sum and a second part, calculates the first part. -/
def firstPart (total second : ℝ) : ℝ := total - second

/-- Calculates simple interest. -/
def simpleInterest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Theorem stating the interest rate for the first part is 3% per annum. -/
theorem interest_rate_first_part (total second : ℝ) (h1 : total = 2704) (h2 : second = 1664) :
  let first := firstPart total second
  let rate2 := 0.05
  let time1 := 8
  let time2 := 3
  simpleInterest first ((3 : ℝ) / 100) time1 = simpleInterest second rate2 time2 := by
  sorry

#check interest_rate_first_part

end interest_rate_first_part_l674_67421


namespace max_product_of_digits_l674_67486

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_product_of_digits (E F G H : ℕ) 
  (hE : is_digit E) (hF : is_digit F) (hG : is_digit G) (hH : is_digit H)
  (distinct : E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H)
  (h_int : ∃ (k : ℕ), E * F = k * (G - H))
  (h_max : ∀ (E' F' G' H' : ℕ), 
    is_digit E' → is_digit F' → is_digit G' → is_digit H' →
    E' ≠ F' ∧ E' ≠ G' ∧ E' ≠ H' ∧ F' ≠ G' ∧ F' ≠ H' ∧ G' ≠ H' →
    (∃ (k' : ℕ), E' * F' = k' * (G' - H')) →
    E * F ≥ E' * F') :
  E * F = 72 :=
sorry

end max_product_of_digits_l674_67486


namespace square_with_semicircles_area_ratio_l674_67416

/-- The ratio of areas for a square with semicircular arcs -/
theorem square_with_semicircles_area_ratio :
  let square_side : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := π * semicircle_radius ^ 2 / 2
  let new_figure_area : ℝ := square_area + 4 * semicircle_area
  new_figure_area / square_area = 1 + π / 2 :=
by sorry

end square_with_semicircles_area_ratio_l674_67416


namespace windshield_wiper_area_l674_67438

/-- The area swept by two semicircular windshield wipers -/
theorem windshield_wiper_area (L : ℝ) (h : L > 0) :
  let area := (2 / 3 * π + Real.sqrt 3 / 4) * L^2
  area = (π * L^2) - ((1 / 3 * π - Real.sqrt 3 / 4) * L^2) :=
by sorry

end windshield_wiper_area_l674_67438


namespace assign_teachers_count_l674_67401

/-- The number of ways to assign 6 teachers to 4 grades -/
def assign_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_grades : ℕ := 4
  let two_specific_teachers : ℕ := 2
  -- Define the function to calculate the number of ways
  sorry

/-- Theorem stating that the number of ways to assign teachers is 240 -/
theorem assign_teachers_count : assign_teachers = 240 := by
  sorry

end assign_teachers_count_l674_67401


namespace max_omega_value_l674_67450

/-- The function f(x) defined in the problem -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

/-- The theorem stating the maximum value of ω -/
theorem max_omega_value (ω φ : ℝ) :
  ω > 0 →
  0 < φ ∧ φ < Real.pi / 2 →
  f ω φ (-Real.pi / 4) = 0 →
  (∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x)) →
  (∀ x y, Real.pi / 18 < x ∧ x < y ∧ y < 2 * Real.pi / 9 → 
    (f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y)) →
  ω ≤ 5 :=
by sorry

end max_omega_value_l674_67450


namespace base_b_is_eight_l674_67436

/-- Given that in base b, the square of 13_b is 211_b, prove that b = 8 -/
theorem base_b_is_eight (b : ℕ) (h : b > 1) :
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 := by
  sorry

end base_b_is_eight_l674_67436


namespace book_arrangement_theorem_l674_67499

theorem book_arrangement_theorem :
  let math_books : ℕ := 4
  let english_books : ℕ := 4
  let group_arrangements : ℕ := 2  -- math books and English books as two groups
  let total_arrangements : ℕ := group_arrangements.factorial * math_books.factorial * english_books.factorial
  total_arrangements = 1152 := by
  sorry

end book_arrangement_theorem_l674_67499


namespace pencils_bought_on_monday_l674_67497

theorem pencils_bought_on_monday (P : ℕ) : P = 20 :=
  by
  -- Define the number of pencils bought on Tuesday
  let tuesday_pencils := 18

  -- Define the number of pencils bought on Wednesday
  let wednesday_pencils := 3 * tuesday_pencils

  -- Define the total number of pencils
  let total_pencils := 92

  -- Assert that the sum of pencils from all days equals the total
  have h : P + tuesday_pencils + wednesday_pencils = total_pencils := by sorry

  -- Prove that P equals 20
  sorry

end pencils_bought_on_monday_l674_67497


namespace largest_n_divisibility_l674_67411

theorem largest_n_divisibility (n : ℕ) : (n + 1) ∣ (n^3 + 10) → n = 0 := by
  sorry

end largest_n_divisibility_l674_67411


namespace apple_distribution_l674_67457

theorem apple_distribution (students : ℕ) (apples : ℕ) : 
  (apples = 4 * students + 3) ∧ 
  (6 * (students - 1) ≤ apples) ∧ 
  (apples ≤ 6 * (students - 1) + 2) →
  (students = 4 ∧ apples = 19) :=
sorry

end apple_distribution_l674_67457
