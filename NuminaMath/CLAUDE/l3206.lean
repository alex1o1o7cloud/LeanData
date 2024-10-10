import Mathlib

namespace arithmetic_sequence_problem_l3206_320612

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 9 + a 27 = 12) : 
  a 13 = 4 := by
sorry

end arithmetic_sequence_problem_l3206_320612


namespace inscribed_circle_diameter_l3206_320673

/-- The diameter of the inscribed circle in a triangle with side lengths 13, 14, and 15 is 8 -/
theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  2 * A / s = 8 := by sorry

end inscribed_circle_diameter_l3206_320673


namespace complex_power_sum_l3206_320675

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * Real.pi / 180) := by
  sorry

end complex_power_sum_l3206_320675


namespace game_result_l3206_320606

def point_function (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def alex_rolls : List Nat := [6, 4, 3, 2, 1]
def bob_rolls : List Nat := [5, 6, 2, 3, 3]

def calculate_points (rolls : List Nat) : Nat :=
  (rolls.map point_function).sum

theorem game_result : 
  (calculate_points alex_rolls) * (calculate_points bob_rolls) = 672 := by
  sorry

end game_result_l3206_320606


namespace zongzi_price_calculation_l3206_320658

theorem zongzi_price_calculation (pork_total red_bean_total : ℕ) 
  (h1 : pork_total = 8000)
  (h2 : red_bean_total = 6000)
  (h3 : ∃ (n : ℕ), n ≠ 0 ∧ pork_total = n * 40 ∧ red_bean_total = n * 30) :
  ∃ (pork_price red_bean_price : ℕ),
    pork_price = 40 ∧
    red_bean_price = 30 ∧
    pork_price = red_bean_price + 10 ∧
    pork_total = red_bean_total + 2000 :=
by
  sorry

end zongzi_price_calculation_l3206_320658


namespace bamboo_nine_sections_l3206_320671

theorem bamboo_nine_sections (a : ℕ → ℚ) (d : ℚ) :
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → a n = a 1 + (n - 1) * d) →
  a 1 + a 2 + a 3 + a 4 = 3 →
  a 7 + a 8 + a 9 = 4 →
  a 1 = 13 / 22 :=
sorry

end bamboo_nine_sections_l3206_320671


namespace sequence_property_l3206_320610

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 101 = 2^5050 := by
sorry

end sequence_property_l3206_320610


namespace geometric_proportion_conclusion_l3206_320625

/-- A set of four real numbers forms a geometric proportion in any order -/
def GeometricProportionAnyOrder (a b c d : ℝ) : Prop :=
  (a / b = c / d ∧ a / b = d / c) ∧
  (a / c = b / d ∧ a / c = d / b) ∧
  (a / d = b / c ∧ a / d = c / b)

/-- The conclusion about four numbers forming a geometric proportion in any order -/
theorem geometric_proportion_conclusion (a b c d : ℝ) 
  (h : GeometricProportionAnyOrder a b c d) :
  (a = b ∧ b = c ∧ c = d) ∨ 
  (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ 
   ((a > 0 ∧ b > 0 ∧ c < 0 ∧ d < 0) ∨
    (a > 0 ∧ c > 0 ∧ b < 0 ∧ d < 0) ∨
    (a > 0 ∧ d > 0 ∧ b < 0 ∧ c < 0) ∨
    (b > 0 ∧ c > 0 ∧ a < 0 ∧ d < 0) ∨
    (b > 0 ∧ d > 0 ∧ a < 0 ∧ c < 0) ∨
    (c > 0 ∧ d > 0 ∧ a < 0 ∧ b < 0))) :=
by sorry

end geometric_proportion_conclusion_l3206_320625


namespace fraction_product_equals_reciprocal_of_2835_l3206_320660

theorem fraction_product_equals_reciprocal_of_2835 :
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) * (1 / 7 : ℚ) = 1 / 2835 := by
  sorry

end fraction_product_equals_reciprocal_of_2835_l3206_320660


namespace jeremy_gives_two_watermelons_l3206_320670

/-- The number of watermelons Jeremy gives to his dad each week. -/
def watermelons_given_to_dad (total_watermelons : ℕ) (weeks_lasted : ℕ) (eaten_per_week : ℕ) : ℕ :=
  (total_watermelons / weeks_lasted) - eaten_per_week

/-- Theorem stating that Jeremy gives 2 watermelons to his dad each week. -/
theorem jeremy_gives_two_watermelons :
  watermelons_given_to_dad 30 6 3 = 2 := by
  sorry

end jeremy_gives_two_watermelons_l3206_320670


namespace initial_trees_count_l3206_320601

theorem initial_trees_count (died cut left : ℕ) 
  (h1 : died = 15)
  (h2 : cut = 23)
  (h3 : left = 48) :
  died + cut + left = 86 := by
  sorry

end initial_trees_count_l3206_320601


namespace range_of_f_l3206_320618

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -21 ∨ y > -21} := by sorry

end range_of_f_l3206_320618


namespace davids_chemistry_marks_l3206_320624

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (total_subjects : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 89)
  (h3 : physics = 82)
  (h4 : biology = 81)
  (h5 : average = 85)
  (h6 : total_subjects = 5) :
  let total_marks := average * total_subjects
  let known_subjects_marks := english + mathematics + physics + biology
  let chemistry := total_marks - known_subjects_marks
  chemistry = 87 := by
    sorry

end davids_chemistry_marks_l3206_320624


namespace willies_cream_calculation_l3206_320604

/-- The amount of whipped cream Willie needs in total (in lbs) -/
def total_cream : ℕ := 300

/-- The amount of cream Willie needs to buy (in lbs) -/
def cream_to_buy : ℕ := 151

/-- The amount of cream Willie got from his farm (in lbs) -/
def cream_from_farm : ℕ := total_cream - cream_to_buy

theorem willies_cream_calculation :
  cream_from_farm = 149 := by
  sorry

end willies_cream_calculation_l3206_320604


namespace fourth_root_of_33177600_l3206_320694

theorem fourth_root_of_33177600 : (33177600 : ℝ) ^ (1/4 : ℝ) = 576 := by sorry

end fourth_root_of_33177600_l3206_320694


namespace smallest_sum_arithmetic_geometric_sequence_l3206_320695

theorem smallest_sum_arithmetic_geometric_sequence 
  (A B C D : ℕ+) 
  (arith_seq : ∃ d : ℤ, (C : ℤ) - (B : ℤ) = d ∧ (B : ℤ) - (A : ℤ) = d)
  (geo_seq : ∃ r : ℚ, (C : ℚ) / (B : ℚ) = r ∧ (D : ℚ) / (C : ℚ) = r)
  (ratio : (C : ℚ) / (B : ℚ) = 7 / 4) :
  (A : ℕ) + B + C + D ≥ 97 ∧ ∃ A' B' C' D' : ℕ+, 
    (∃ d : ℤ, (C' : ℤ) - (B' : ℤ) = d ∧ (B' : ℤ) - (A' : ℤ) = d) ∧
    (∃ r : ℚ, (C' : ℚ) / (B' : ℚ) = r ∧ (D' : ℚ) / (C' : ℚ) = r) ∧
    (C' : ℚ) / (B' : ℚ) = 7 / 4 ∧
    (A' : ℕ) + B' + C' + D' = 97 :=
by sorry

end smallest_sum_arithmetic_geometric_sequence_l3206_320695


namespace add_3_15_base6_l3206_320674

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to its decimal representation --/
def toDecimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to its base 6 representation --/
def toBase6 (n : Nat) : Base6 :=
  sorry

/-- Addition in base 6 --/
def addBase6 (a b : Base6) : Base6 :=
  toBase6 (toDecimal a + toDecimal b)

theorem add_3_15_base6 :
  addBase6 (toBase6 3) (toBase6 15) = toBase6 22 := by
  sorry

end add_3_15_base6_l3206_320674


namespace two_sessions_scientific_notation_l3206_320626

theorem two_sessions_scientific_notation :
  78200000000 = 7.82 * (10 : ℝ)^10 := by sorry

end two_sessions_scientific_notation_l3206_320626


namespace distinct_primes_dividing_P_l3206_320672

def P : ℕ := (List.range 10).foldl (· * ·) 1

theorem distinct_primes_dividing_P :
  (Finset.filter (fun p => Nat.Prime p ∧ P % p = 0) (Finset.range 11)).card = 4 := by
  sorry

end distinct_primes_dividing_P_l3206_320672


namespace quotient_calculation_l3206_320633

theorem quotient_calculation (divisor dividend remainder quotient : ℕ) : 
  divisor = 17 → dividend = 76 → remainder = 8 → quotient = 4 →
  dividend = divisor * quotient + remainder :=
by
  sorry

end quotient_calculation_l3206_320633


namespace remaining_quarters_l3206_320682

-- Define the initial amount, total spent, and value of a quarter
def initial_amount : ℚ := 40
def total_spent : ℚ := 32.25
def quarter_value : ℚ := 0.25

-- Theorem to prove
theorem remaining_quarters : 
  (initial_amount - total_spent) / quarter_value = 31 := by
  sorry

end remaining_quarters_l3206_320682


namespace equation_solution_l3206_320683

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) = 1) ∧ x = 2 := by
  sorry

end equation_solution_l3206_320683


namespace tan_product_zero_l3206_320602

theorem tan_product_zero (a b : ℝ) 
  (h : 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) : 
  Real.tan (a / 2) * Real.tan (b / 2) = 0 := by
sorry

end tan_product_zero_l3206_320602


namespace max_type_B_bins_l3206_320603

def unit_price_A : ℕ := 300
def unit_price_B : ℕ := 450
def total_budget : ℕ := 8000
def total_bins : ℕ := 20

theorem max_type_B_bins :
  ∀ y : ℕ,
    y ≤ 13 ∧
    y ≤ total_bins ∧
    unit_price_A * (total_bins - y) + unit_price_B * y ≤ total_budget ∧
    (∀ z : ℕ, z > y →
      z > 13 ∨
      z > total_bins ∨
      unit_price_A * (total_bins - z) + unit_price_B * z > total_budget) :=
by sorry

end max_type_B_bins_l3206_320603


namespace projections_on_concentric_circles_imply_parallelogram_l3206_320643

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Projection of a point onto a line segment -/
def project (p : Point) (a b : Point) : Point :=
  sorry

/-- Check if four points form an inscribed quadrilateral in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

/-- Check if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Main theorem -/
theorem projections_on_concentric_circles_imply_parallelogram 
  (q : Quadrilateral) (p1 p2 : Point) (c1 c2 : Circle) :
  c1.center = c2.center →
  c1.radius ≠ c2.radius →
  is_inscribed (Quadrilateral.mk 
    (project p1 q.a q.b) (project p1 q.b q.c) 
    (project p1 q.c q.d) (project p1 q.d q.a)) c1 →
  is_inscribed (Quadrilateral.mk 
    (project p2 q.a q.b) (project p2 q.b q.c) 
    (project p2 q.c q.d) (project p2 q.d q.a)) c2 →
  is_parallelogram q :=
sorry

end projections_on_concentric_circles_imply_parallelogram_l3206_320643


namespace quadratic_equation_root_l3206_320646

theorem quadratic_equation_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x => p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f (-1) = 0) →
  (f (-r * (p - q) / (p * (q - r))) = 0) :=
by sorry

end quadratic_equation_root_l3206_320646


namespace systematic_sampling_third_event_l3206_320699

/-- Given a total of 960 students, selecting every 30th student starting from
    student number 30, the number of selected students in the interval [701, 960] is 9. -/
theorem systematic_sampling_third_event (total_students : Nat) (selection_interval : Nat) 
    (first_selected : Nat) (event_start : Nat) (event_end : Nat) : Nat :=
  have h1 : total_students = 960 := by sorry
  have h2 : selection_interval = 30 := by sorry
  have h3 : first_selected = 30 := by sorry
  have h4 : event_start = 701 := by sorry
  have h5 : event_end = 960 := by sorry
  9

#check systematic_sampling_third_event

end systematic_sampling_third_event_l3206_320699


namespace circle_equation_l3206_320685

/-- The circle passing through points A(-1, 1) and B(-2, -2), with center C lying on the line x+y-1=0, has the standard equation (x - 3)² + (y + 2)² = 25 -/
theorem circle_equation : 
  ∀ (C : ℝ × ℝ) (r : ℝ),
  (C.1 + C.2 - 1 = 0) →  -- Center C lies on the line x+y-1=0
  ((-1 - C.1)^2 + (1 - C.2)^2 = r^2) →  -- Circle passes through A(-1, 1)
  ((-2 - C.1)^2 + (-2 - C.2)^2 = r^2) →  -- Circle passes through B(-2, -2)
  ∀ (x y : ℝ), 
  ((x - 3)^2 + (y + 2)^2 = 25) ↔ ((x - C.1)^2 + (y - C.2)^2 = r^2) :=
by sorry

end circle_equation_l3206_320685


namespace solve_for_m_l3206_320666

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 := by
  sorry

end solve_for_m_l3206_320666


namespace complex_arithmetic_equality_l3206_320644

theorem complex_arithmetic_equality : -6 / 2 + (1/3 - 3/4) * 12 + (-3)^2 = 1 := by
  sorry

end complex_arithmetic_equality_l3206_320644


namespace base8_to_base10_conversion_l3206_320617

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of 246₈ -/
def base8Number : List Nat := [6, 4, 2]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 166 := by
  sorry

end base8_to_base10_conversion_l3206_320617


namespace simplify_expression_l3206_320698

theorem simplify_expression : ((-Real.sqrt 3)^2)^(-1/2 : ℝ) = Real.sqrt 3 / 3 := by sorry

end simplify_expression_l3206_320698


namespace table_price_is_56_l3206_320650

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of some chairs and 2 tables -/
axiom price_ratio : ∃ x : ℝ, 2 * chair_price + table_price = 0.6 * (x * chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end table_price_is_56_l3206_320650


namespace sum_of_powers_l3206_320684

theorem sum_of_powers : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end sum_of_powers_l3206_320684


namespace wire_length_is_250_meters_l3206_320623

-- Define the density of copper
def copper_density : Real := 8900

-- Define the volume of wire bought by Chek
def wire_volume : Real := 0.5e-3

-- Define the diagonal of the wire's square cross-section
def wire_diagonal : Real := 2e-3

-- Theorem to prove
theorem wire_length_is_250_meters :
  let cross_section_area := (wire_diagonal ^ 2) / 2
  let wire_length := wire_volume / cross_section_area
  wire_length = 250 := by sorry

end wire_length_is_250_meters_l3206_320623


namespace kara_book_count_l3206_320662

/-- The number of books read by each person in the Book Tournament --/
structure BookCount where
  candice : ℕ
  amanda : ℕ
  kara : ℕ
  patricia : ℕ

/-- The conditions of the Book Tournament --/
def BookTournament (bc : BookCount) : Prop :=
  bc.candice = 18 ∧
  bc.candice = 3 * bc.amanda ∧
  bc.kara = bc.amanda / 2

theorem kara_book_count (bc : BookCount) (h : BookTournament bc) : bc.kara = 3 := by
  sorry

end kara_book_count_l3206_320662


namespace robins_walk_distance_l3206_320609

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_walk_distance (house_to_center : ℕ) (initial_walk : ℕ) : 
  house_to_center = 500 →
  initial_walk = 200 →
  initial_walk + initial_walk + house_to_center = 900 := by
  sorry

end robins_walk_distance_l3206_320609


namespace expand_and_simplify_simplify_complex_fraction_l3206_320613

-- Problem 1
theorem expand_and_simplify (x y : ℝ) : 
  (x + y) * (x - y) + y * (y - 2) = x^2 - 2*y := by sorry

-- Problem 2
theorem simplify_complex_fraction (m : ℝ) (hm2 : m ≠ 2) (hm_2 : m ≠ -2) : 
  (1 - m / (m + 2)) / ((m^2 - 4*m + 4) / (m^2 - 4)) = 2 / (m - 2) := by sorry

end expand_and_simplify_simplify_complex_fraction_l3206_320613


namespace magnitude_of_complex_number_l3206_320688

theorem magnitude_of_complex_number (z : ℂ) : z = (4 - 2*I) / (1 + I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_complex_number_l3206_320688


namespace parallel_planes_imply_parallel_lines_l3206_320648

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (a b : Line) (α β : Plane) 
  (ha : subset a α) (hb : subset b α) :
  parallel α β → (parallel α β ∧ parallel α β) := by
  sorry

end parallel_planes_imply_parallel_lines_l3206_320648


namespace initial_average_customers_l3206_320655

theorem initial_average_customers (x : ℕ) (today_customers : ℕ) (new_average : ℕ) 
  (h1 : x = 1)
  (h2 : today_customers = 120)
  (h3 : new_average = 90)
  : ∃ initial_average : ℕ, initial_average = 60 ∧ 
    (initial_average * x + today_customers) / (x + 1) = new_average :=
by
  sorry

end initial_average_customers_l3206_320655


namespace school_boys_count_l3206_320628

theorem school_boys_count :
  ∀ (total_students : ℕ) (boys : ℕ),
    total_students = 400 →
    boys + (boys * total_students / 100) = total_students →
    boys = 80 :=
by
  sorry

end school_boys_count_l3206_320628


namespace tempo_insured_fraction_l3206_320653

/-- Represents the insurance details of a tempo --/
structure TempoInsurance where
  premium_rate : Rat
  premium_amount : Rat
  original_value : Rat

/-- Calculates the fraction of the original value that is insured --/
def insured_fraction (insurance : TempoInsurance) : Rat :=
  (insurance.premium_amount / insurance.premium_rate) / insurance.original_value

/-- Theorem stating that for the given insurance details, the insured fraction is 5/7 --/
theorem tempo_insured_fraction :
  let insurance : TempoInsurance := {
    premium_rate := 3 / 100,
    premium_amount := 300,
    original_value := 14000
  }
  insured_fraction insurance = 5 / 7 := by sorry

end tempo_insured_fraction_l3206_320653


namespace inequality_proof_l3206_320681

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end inequality_proof_l3206_320681


namespace carbon_monoxide_weight_l3206_320678

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (c o : ℝ) : ℝ := c + o

/-- Theorem: The molecular weight of Carbon monoxide (CO) is 28.01 g/mol -/
theorem carbon_monoxide_weight : molecular_weight carbon_weight oxygen_weight = 28.01 := by
  sorry

end carbon_monoxide_weight_l3206_320678


namespace solve_for_x_l3206_320621

theorem solve_for_x (x : ℝ) : 
  let M := 2*x - 2
  let N := 2*x + 3
  2*M - N = 1 → x = 4 := by
sorry

end solve_for_x_l3206_320621


namespace smallest_four_digit_divisible_by_53_l3206_320629

theorem smallest_four_digit_divisible_by_53 :
  ∃ n : ℕ, n = 1007 ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 53 ∣ m → n ≤ m) ∧
  1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n :=
by sorry

end smallest_four_digit_divisible_by_53_l3206_320629


namespace negative_product_implies_positive_fraction_l3206_320605

theorem negative_product_implies_positive_fraction
  (x y z : ℝ) (h : x * y^3 * z^2 < 0) (hy : y ≠ 0) :
  -(x^3 * z^4) / y^5 > 0 :=
by sorry

end negative_product_implies_positive_fraction_l3206_320605


namespace sqrt_320_simplification_l3206_320686

theorem sqrt_320_simplification : Real.sqrt 320 = 8 * Real.sqrt 5 := by
  sorry

end sqrt_320_simplification_l3206_320686


namespace square_plus_inverse_square_l3206_320692

theorem square_plus_inverse_square (a : ℝ) (h : a - (1 / a) = 5) : a^2 + (1 / a^2) = 27 := by
  sorry

end square_plus_inverse_square_l3206_320692


namespace sqrt_equation_solution_l3206_320679

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end sqrt_equation_solution_l3206_320679


namespace marble_redistribution_l3206_320691

theorem marble_redistribution (dilan martha phillip veronica : ℕ) 
  (h1 : dilan = 14)
  (h2 : martha = 20)
  (h3 : phillip = 19)
  (h4 : veronica = 7) :
  (dilan + martha + phillip + veronica) / 4 = 15 := by
  sorry

end marble_redistribution_l3206_320691


namespace beta_max_success_ratio_l3206_320608

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ) :
  alpha_day1_score = 210 →
  alpha_day1_total = 400 →
  alpha_day2_score = 210 →
  alpha_day2_total = 300 →
  beta_day1_total + beta_day2_total = 700 →
  beta_day1_total < 400 →
  beta_day2_total < 400 →
  beta_day1_score > 0 →
  beta_day2_score > 0 →
  (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total →
  (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total →
  (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5 →
  (beta_day1_score + beta_day2_score : ℚ) / 700 ≤ 139/700 :=
by sorry

end beta_max_success_ratio_l3206_320608


namespace apple_juice_cost_l3206_320635

def orange_juice_cost : ℚ := 70 / 100
def total_bottles : ℕ := 70
def total_cost : ℚ := 4620 / 100
def orange_juice_bottles : ℕ := 42

theorem apple_juice_cost :
  let apple_juice_bottles : ℕ := total_bottles - orange_juice_bottles
  let apple_juice_total_cost : ℚ := total_cost - (orange_juice_cost * orange_juice_bottles)
  apple_juice_total_cost / apple_juice_bottles = 60 / 100 :=
by sorry

end apple_juice_cost_l3206_320635


namespace max_notebooks_purchasable_l3206_320657

def total_money : ℚ := 30
def notebook_cost : ℚ := 2.4

theorem max_notebooks_purchasable :
  ⌊total_money / notebook_cost⌋ = 12 := by sorry

end max_notebooks_purchasable_l3206_320657


namespace johnson_family_reunion_l3206_320697

theorem johnson_family_reunion (children : ℕ) (adults : ℕ) (blue_adults : ℕ) : 
  children = 45 →
  adults = children / 3 →
  blue_adults = adults / 3 →
  adults - blue_adults = 10 := by
sorry

end johnson_family_reunion_l3206_320697


namespace largest_common_number_l3206_320615

def first_sequence (n : ℕ) : ℤ := 5 + 8 * (n - 1)

def second_sequence (n : ℕ) : ℤ := 3 + 9 * (n - 1)

def is_common (x : ℤ) : Prop :=
  ∃ (n m : ℕ), first_sequence n = x ∧ second_sequence m = x

theorem largest_common_number :
  ∃ (x : ℤ), is_common x ∧ x ≤ 150 ∧
  ∀ (y : ℤ), is_common y ∧ y ≤ 150 → y ≤ x ∧
  x = 93 :=
sorry

end largest_common_number_l3206_320615


namespace unique_special_function_l3206_320622

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f ((x + 2) + 2) = f (x + 2)) ∧  -- g(x) = f(x+2) is even
  (∀ x, x ∈ Set.Icc 0 2 → f x = x)  -- f(x) = x for x ∈ [0, 2]

/-- There exists a unique function satisfying the special_function conditions -/
theorem unique_special_function : ∃! f : ℝ → ℝ, special_function f :=
sorry

end unique_special_function_l3206_320622


namespace jen_jam_consumption_l3206_320659

theorem jen_jam_consumption (total_jam : ℚ) : 
  let lunch_consumption := (1 : ℚ) / 3
  let after_lunch := total_jam - lunch_consumption * total_jam
  let after_dinner := (4 : ℚ) / 7 * total_jam
  let dinner_consumption := (after_lunch - after_dinner) / after_lunch
  dinner_consumption = (1 : ℚ) / 7 := by sorry

end jen_jam_consumption_l3206_320659


namespace degrees_to_radians_210_l3206_320676

theorem degrees_to_radians_210 : 
  (210 : ℝ) * (π / 180) = (7 * π) / 6 := by
  sorry

end degrees_to_radians_210_l3206_320676


namespace unique_valid_number_l3206_320668

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 10 = (n / 10) % 10 ∧
  (n / 100) % 10 = n / 1000 ∧
  ∃ k : ℕ, n = k * k

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 7744 :=
sorry

end unique_valid_number_l3206_320668


namespace bobs_assorted_candies_l3206_320645

/-- The problem of calculating Bob's assorted candies -/
theorem bobs_assorted_candies 
  (total_candies : ℕ) 
  (chewing_gums : ℕ) 
  (chocolate_bars : ℕ) 
  (h1 : total_candies = 50)
  (h2 : chewing_gums = 15)
  (h3 : chocolate_bars = 20) :
  total_candies - (chewing_gums + chocolate_bars) = 15 :=
by sorry

end bobs_assorted_candies_l3206_320645


namespace kyle_monthly_income_l3206_320636

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries_eating_out : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350
def gas_maintenance : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance

theorem kyle_monthly_income : total_expenses = 3200 := by
  sorry

end kyle_monthly_income_l3206_320636


namespace fraction_order_l3206_320631

theorem fraction_order : 
  let f1 := 18 / 14
  let f2 := 16 / 12
  let f3 := 20 / 16
  5 / 4 < f1 ∧ f1 < f2 ∧ f3 < f1 := by sorry

end fraction_order_l3206_320631


namespace jonah_aquarium_fish_count_l3206_320611

/-- Calculates the final number of fish in Jonah's aquarium after a series of events. -/
def final_fish_count (initial : ℕ) (added : ℕ) (eaten : ℕ) (returned : ℕ) (exchanged : ℕ) : ℕ :=
  initial + added - eaten - returned + exchanged

/-- Theorem stating that given the initial conditions and series of events, 
    the final number of fish in Jonah's aquarium is 11. -/
theorem jonah_aquarium_fish_count : 
  final_fish_count 14 2 6 2 3 = 11 := by sorry

end jonah_aquarium_fish_count_l3206_320611


namespace shopping_tax_rate_l3206_320638

def shopping_problem (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ) 
                     (other_tax_rate : ℝ) (total_tax_rate : ℝ) : Prop :=
  clothing_percent + food_percent + other_percent = 100 ∧
  clothing_percent = 50 ∧
  food_percent = 20 ∧
  other_percent = 30 ∧
  other_tax_rate = 10 ∧
  total_tax_rate = 5 ∧
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = 
    total_tax_rate * 100 ∧
    clothing_tax_rate = 4

theorem shopping_tax_rate :
  ∀ (clothing_percent food_percent other_percent other_tax_rate total_tax_rate : ℝ),
  shopping_problem clothing_percent food_percent other_percent other_tax_rate total_tax_rate →
  ∃ (clothing_tax_rate : ℝ), clothing_tax_rate = 4 :=
by
  sorry

#check shopping_tax_rate

end shopping_tax_rate_l3206_320638


namespace village_population_equality_l3206_320661

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_increase + x_decrease)

theorem village_population_equality :
  years_until_equal_population 78000 1200 42000 800 = 18 := by
  sorry

end village_population_equality_l3206_320661


namespace complex_exponential_sum_l3206_320600

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + Complex.I * (1 / 2 : ℂ) →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - Complex.I * (1 / 2 : ℂ) :=
by sorry

end complex_exponential_sum_l3206_320600


namespace proportional_segments_l3206_320637

theorem proportional_segments (a b c d : ℝ) : 
  a / b = c / d → a = 2 → b = 4 → c = 3 → d = 6 := by
  sorry

end proportional_segments_l3206_320637


namespace count_pairs_satisfying_inequality_l3206_320634

def S : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3}

theorem count_pairs_satisfying_inequality :
  (Finset.filter (fun p : ℤ × ℤ => 
    p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ p.2^2 < (5/4) * p.1^2)
    (S.product S)).card = 18 := by
  sorry

end count_pairs_satisfying_inequality_l3206_320634


namespace inequality_proof_l3206_320640

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / ((b + c)^2 + a^2) +
  (c + a - b)^2 / ((c + a)^2 + b^2) +
  (a + b - c)^2 / ((a + b)^2 + c^2) ≥ 3/5 := by
  sorry

end inequality_proof_l3206_320640


namespace triangle_abc_proof_l3206_320614

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a / (b * c) + c / (a * b) - b / (a * c) = 1 / (a * Real.cos C + c * Real.cos A)) →
  (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2) →
  (b / Real.sin B = 2 * Real.sqrt 3) →
  (c > a) →
  (B = π / 3 ∧ c = 2 * Real.sqrt 3) :=
by sorry

end triangle_abc_proof_l3206_320614


namespace product_of_sum_of_squares_l3206_320649

theorem product_of_sum_of_squares (a b n k : ℝ) :
  let K := a^2 + b^2
  let P := n^2 + k^2
  K * P = (a*n + b*k)^2 + (a*k - b*n)^2 := by
  sorry

end product_of_sum_of_squares_l3206_320649


namespace condition_sufficient_not_necessary_l3206_320693

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  ¬(∀ x : ℝ, x > -1 → (x + 1) * (x - 3) < 0) :=
by sorry

end condition_sufficient_not_necessary_l3206_320693


namespace math_voters_l3206_320677

theorem math_voters (total_students : ℕ) (math_percentage : ℚ) : 
  total_students = 480 → math_percentage = 40 / 100 →
  (math_percentage * total_students.cast) = 192 := by
sorry

end math_voters_l3206_320677


namespace width_covered_formula_l3206_320651

/-- The width covered by n asbestos tiles -/
def width_covered (n : ℕ+) : ℝ :=
  let tile_width : ℝ := 60
  let overlap : ℝ := 10
  (n : ℝ) * (tile_width - overlap) + overlap

/-- Theorem: The width covered by n asbestos tiles is (50n + 10) cm -/
theorem width_covered_formula (n : ℕ+) :
  width_covered n = 50 * (n : ℝ) + 10 := by
  sorry

end width_covered_formula_l3206_320651


namespace function_extrema_m_range_l3206_320689

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

-- State the theorem
theorem function_extrema_m_range (m : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ (x : ℝ), f m x ≤ f m x_max ∧ f m x_min ≤ f m x) →
  m < -3 ∨ m > 6 :=
sorry

end function_extrema_m_range_l3206_320689


namespace price_difference_proof_l3206_320654

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.25

def amy_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate)
def bob_total : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)
def carla_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference_proof :
  carla_total - amy_total = 6.744 ∧ carla_total - bob_total = 6.744 :=
by sorry

end price_difference_proof_l3206_320654


namespace workday_meeting_percentage_l3206_320642

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes

/-- Calculates the total minutes in a workday -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem stating that the percentage of workday spent in meetings is 40% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 40 := by
  sorry

end workday_meeting_percentage_l3206_320642


namespace largest_possible_reflections_l3206_320619

/-- Represents the angle of reflection at each point -/
def reflection_angle (n : ℕ) : ℝ := 15 * n

/-- The condition for the beam to hit perpendicularly and retrace its path -/
def valid_reflection (n : ℕ) : Prop := reflection_angle n ≤ 90

theorem largest_possible_reflections : ∃ (max_n : ℕ), 
  (∀ n : ℕ, valid_reflection n → n ≤ max_n) ∧ 
  valid_reflection max_n ∧ 
  max_n = 6 :=
sorry

end largest_possible_reflections_l3206_320619


namespace total_driving_time_bound_l3206_320696

/-- Represents the driving scenario with given distances and speeds -/
structure DrivingScenario where
  distance_first : ℝ
  time_first : ℝ
  distance_second : ℝ
  distance_third : ℝ
  distance_fourth : ℝ
  speed_second : ℝ
  speed_third : ℝ
  speed_fourth : ℝ

/-- The total driving time is less than or equal to 10 hours -/
theorem total_driving_time_bound (scenario : DrivingScenario) 
  (h1 : scenario.distance_first = 120)
  (h2 : scenario.time_first = 3)
  (h3 : scenario.distance_second = 60)
  (h4 : scenario.distance_third = 90)
  (h5 : scenario.distance_fourth = 200)
  (h6 : scenario.speed_second > 0)
  (h7 : scenario.speed_third > 0)
  (h8 : scenario.speed_fourth > 0) :
  scenario.time_first + 
  scenario.distance_second / scenario.speed_second + 
  scenario.distance_third / scenario.speed_third + 
  scenario.distance_fourth / scenario.speed_fourth ≤ 10 := by
  sorry

end total_driving_time_bound_l3206_320696


namespace probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l3206_320656

-- Define the probabilities for complaints in a single month
def p_zero_complaints : ℝ := 0.3
def p_one_complaint : ℝ := 0.5
def p_two_complaints : ℝ := 0.2

-- Theorem for part (I)
theorem probability_no_more_than_one_complaint :
  p_zero_complaints + p_one_complaint = 0.8 := by sorry

-- Theorem for part (II)
theorem probability_two_complaints_in_two_months :
  let p_two_total := p_zero_complaints * p_two_complaints +
                     p_two_complaints * p_zero_complaints +
                     p_one_complaint * p_one_complaint
  p_two_total = 0.37 := by sorry

end probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l3206_320656


namespace problem_solution_l3206_320647

theorem problem_solution (a b : ℝ) (h1 : a * b = 7) (h2 : a - b = 5) :
  a^2 - 6*a*b + b^2 = -3 := by
  sorry

end problem_solution_l3206_320647


namespace triangle_problem_l3206_320630

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.B - t.A = π / 6) 
  (h4 : t.A + t.B + t.C = π) -- Triangle angle sum
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of sines
  (h6 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of sines
  : Real.sin t.A = Real.sqrt 7 / 14 ∧ t.c = (11 / 7) * Real.sqrt 7 := by
  sorry

end triangle_problem_l3206_320630


namespace percentage_calculation_l3206_320663

theorem percentage_calculation (p : ℝ) : 
  0.25 * 900 = p / 100 * 1600 - 15 → p = 1500 := by
sorry

end percentage_calculation_l3206_320663


namespace lcm_one_to_ten_l3206_320669

theorem lcm_one_to_ten : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := by
  sorry

#eval Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

end lcm_one_to_ten_l3206_320669


namespace sum_first_seven_primes_mod_eighth_prime_l3206_320680

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum % eighth_prime) = 1 := by
  sorry

end sum_first_seven_primes_mod_eighth_prime_l3206_320680


namespace shaded_area_between_squares_l3206_320687

/-- The area of the shaded region in a figure with two concentric squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10) 
  (h2 : small_side = 4) 
  (h3 : large_side > small_side) : 
  (large_side^2 - small_side^2) / 4 = 21 := by
  sorry

end shaded_area_between_squares_l3206_320687


namespace foci_coincide_l3206_320607

/-- The value of m for which the foci of the given hyperbola and ellipse coincide -/
theorem foci_coincide (m : ℝ) : 
  (∀ x y : ℝ, y^2/2 - x^2/m = 1 ↔ (y^2/2 = 1 + x^2/m)) ∧ 
  (∀ x y : ℝ, x^2/4 + y^2/9 = 1) ∧
  (∃ c : ℝ, c^2 = 2 + m ∧ c^2 = 5) →
  m = 3 := by
sorry

end foci_coincide_l3206_320607


namespace horner_method_v4_l3206_320632

def horner_polynomial (x : ℝ) : ℝ := 1 + 8*x + 7*x^2 + 5*x^4 + 4*x^5 + 3*x^6

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 4
  let v2 := v1 * x + 5
  let v3 := v2 * x + 0
  v3 * x + 7

theorem horner_method_v4 :
  horner_v4 5 = 2507 :=
by sorry

end horner_method_v4_l3206_320632


namespace min_value_a_a_range_l3206_320639

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 3|

-- Theorem 1
theorem min_value_a (a : ℝ) :
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = 2 ∨ a = -8 := by
  sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 0 → f x a ≤ |x - 4|) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end min_value_a_a_range_l3206_320639


namespace theft_culprits_l3206_320665

-- Define the guilt status of each person
variable (E F G : Prop)

-- E represents "Elise is guilty"
-- F represents "Fred is guilty"
-- G represents "Gaétan is guilty"

-- Define the given conditions
axiom cond1 : ¬G → F
axiom cond2 : ¬E → G
axiom cond3 : G → E
axiom cond4 : E → ¬F

-- Theorem to prove
theorem theft_culprits : E ∧ G ∧ ¬F := by
  sorry

end theft_culprits_l3206_320665


namespace unique_rectangle_arrangement_l3206_320664

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.height)

/-- Checks if two rectangles have equal perimeters -/
def equalPerimeters (r1 r2 : Rectangle) : Prop := r1.perimeter = r2.perimeter

/-- Checks if the total area of two rectangles is 81 -/
def totalAreaIs81 (r1 r2 : Rectangle) : Prop := r1.area + r2.area = 81

/-- The main theorem stating that the only way to arrange 81 unit squares into two rectangles
    with equal perimeters is to form rectangles with dimensions 3 × 11 and 6 × 8 -/
theorem unique_rectangle_arrangement :
  ∀ r1 r2 : Rectangle,
    equalPerimeters r1 r2 → totalAreaIs81 r1 r2 →
    ((r1.width = 3 ∧ r1.height = 11) ∧ (r2.width = 6 ∧ r2.height = 8)) ∨
    ((r1.width = 6 ∧ r1.height = 8) ∧ (r2.width = 3 ∧ r2.height = 11)) := by
  sorry

end unique_rectangle_arrangement_l3206_320664


namespace scientific_notation_equality_l3206_320620

/-- Proves that 448000 is equal to 4.48 * 10^5 in scientific notation -/
theorem scientific_notation_equality : 448000 = 4.48 * (10 ^ 5) := by
  sorry

end scientific_notation_equality_l3206_320620


namespace distance_inequality_l3206_320627

theorem distance_inequality (a : ℝ) :
  (abs (a - 1) < 3) → (-2 < a ∧ a < 4) := by
  sorry

end distance_inequality_l3206_320627


namespace road_repair_group_size_l3206_320667

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end road_repair_group_size_l3206_320667


namespace twins_age_problem_l3206_320616

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 17 → age = 8 :=
by sorry

end twins_age_problem_l3206_320616


namespace cubic_function_continuous_l3206_320641

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- State the theorem that f is continuous for all real x
theorem cubic_function_continuous :
  ∀ x : ℝ, ContinuousAt f x :=
by
  sorry

end cubic_function_continuous_l3206_320641


namespace franklin_students_count_l3206_320690

/-- The number of Valentines Mrs. Franklin already has -/
def valentines_owned : ℝ := 58.0

/-- The number of additional Valentines Mrs. Franklin needs -/
def valentines_needed : ℝ := 16.0

/-- The number of students Mrs. Franklin has -/
def number_of_students : ℝ := valentines_owned + valentines_needed

theorem franklin_students_count : number_of_students = 74.0 := by
  sorry

end franklin_students_count_l3206_320690


namespace tim_and_tina_same_age_l3206_320652

def tim_age_condition (x : ℕ) : Prop := x + 2 = 2 * (x - 2)

def tina_age_condition (y : ℕ) : Prop := y + 3 = 3 * (y - 3)

theorem tim_and_tina_same_age :
  ∃ (x y : ℕ), tim_age_condition x ∧ tina_age_condition y ∧ x = y :=
by
  sorry

end tim_and_tina_same_age_l3206_320652
