import Mathlib

namespace exactly_three_sets_sum_to_30_l1662_166223

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a ConsecutiveSet summing to 30 -/
def sums_to_30 (s : ConsecutiveSet) : Prop :=
  sum_consecutive_set s = 30

theorem exactly_three_sets_sum_to_30 :
  ∃! (sets : Finset ConsecutiveSet), 
    Finset.card sets = 3 ∧ 
    (∀ s ∈ sets, sums_to_30 s) ∧
    (∀ s : ConsecutiveSet, sums_to_30 s → s ∈ sets) :=
sorry

end exactly_three_sets_sum_to_30_l1662_166223


namespace possible_values_l1662_166211

def Rectangle := Fin 3 → Fin 4 → ℕ

def valid_rectangle (r : Rectangle) : Prop :=
  (∀ i j, r i j ∈ Finset.range 13) ∧
  (∀ i j k, i ≠ j → r i k ≠ r j k) ∧
  (∀ k, r 0 k + r 1 k = 2 * r 2 k) ∧
  (r 0 0 = 6 ∧ r 1 0 = 4 ∧ r 2 1 = 8 ∧ r 2 2 = 11)

theorem possible_values (r : Rectangle) (h : valid_rectangle r) :
  r 2 3 = 2 ∨ r 2 3 = 11 :=
sorry

end possible_values_l1662_166211


namespace completing_square_l1662_166260

theorem completing_square (x : ℝ) : x^2 - 4*x = 6 ↔ (x - 2)^2 = 10 := by
  sorry

end completing_square_l1662_166260


namespace shelter_blocks_count_l1662_166278

/-- Calculates the number of blocks needed for a rectangular shelter --/
def shelter_blocks (length width height : ℕ) : ℕ :=
  let exterior_volume := length * width * height
  let interior_length := length - 2
  let interior_width := width - 2
  let interior_height := height - 2
  let interior_volume := interior_length * interior_width * interior_height
  exterior_volume - interior_volume

/-- Proves that the number of blocks for a shelter with given dimensions is 528 --/
theorem shelter_blocks_count :
  shelter_blocks 14 12 6 = 528 := by
  sorry

end shelter_blocks_count_l1662_166278


namespace daisy_percentage_in_bouquet_l1662_166218

theorem daisy_percentage_in_bouquet : 
  ∀ (total_flowers : ℕ) (white_flowers yellow_flowers white_tulips yellow_tulips white_daisies yellow_daisies : ℕ),
  total_flowers > 0 →
  white_flowers + yellow_flowers = total_flowers →
  white_tulips + white_daisies = white_flowers →
  yellow_tulips + yellow_daisies = yellow_flowers →
  white_tulips = white_flowers / 2 →
  yellow_daisies = (2 * yellow_flowers) / 3 →
  white_flowers = (7 * total_flowers) / 10 →
  (white_daisies + yellow_daisies) * 100 = 55 * total_flowers :=
by sorry

end daisy_percentage_in_bouquet_l1662_166218


namespace school_ball_purchase_l1662_166274

-- Define the unit prices
def soccer_price : ℝ := 40
def basketball_price : ℝ := 60

-- Define the total number of balls and max cost
def total_balls : ℕ := 200
def max_cost : ℝ := 9600

-- Theorem statement
theorem school_ball_purchase :
  -- Condition 1: Basketball price is 20 more than soccer price
  (basketball_price = soccer_price + 20) →
  -- Condition 2: Cost ratio of basketballs to soccer balls
  (6000 / basketball_price = 1.25 * (3200 / soccer_price)) →
  -- Condition 3 and 4 are implicitly used in the conclusion
  -- Conclusion: Correct prices and minimum number of soccer balls
  (soccer_price = 40 ∧ 
   basketball_price = 60 ∧ 
   ∀ m : ℕ, (m : ℝ) * soccer_price + (total_balls - m : ℝ) * basketball_price ≤ max_cost → m ≥ 120) :=
by sorry

end school_ball_purchase_l1662_166274


namespace class_composition_l1662_166257

/-- Represents a pair of numbers reported by a student -/
structure ReportedPair :=
  (classmates : ℕ)
  (female_classmates : ℕ)

/-- Checks if a reported pair is valid given the actual numbers of boys and girls -/
def is_valid_report (report : ReportedPair) (boys girls : ℕ) : Prop :=
  (report.classmates = boys + girls - 1 ∧ (report.female_classmates = girls ∨ report.female_classmates = girls + 2 ∨ report.female_classmates = girls - 2)) ∨
  (report.female_classmates = girls ∧ (report.classmates = boys + girls - 1 + 2 ∨ report.classmates = boys + girls - 1 - 2))

theorem class_composition 
  (reports : List ReportedPair)
  (h1 : (12, 18) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h2 : (15, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h3 : (11, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h4 : ∀ r ∈ reports, is_valid_report r 13 16) :
  ∃ (boys girls : ℕ), boys = 13 ∧ girls = 16 ∧ 
    (∀ r ∈ reports, is_valid_report r boys girls) :=
by
  sorry

end class_composition_l1662_166257


namespace zoo_visitors_l1662_166245

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) :
  num_cars = 3.0 → people_per_car = 63.0 → num_cars * people_per_car = 189.0 := by
  sorry

end zoo_visitors_l1662_166245


namespace division_result_l1662_166214

theorem division_result : 75 / 0.05 = 1500 := by
  sorry

end division_result_l1662_166214


namespace sum_of_positive_numbers_l1662_166224

theorem sum_of_positive_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y = 30 → x * z = 60 → y * z = 90 → 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end sum_of_positive_numbers_l1662_166224


namespace trig_simplification_l1662_166292

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  1/2 * (1 - Real.cos x ^ 2 - Real.cos y ^ 2) := by
  sorry

end trig_simplification_l1662_166292


namespace binary_multiplication_and_shift_l1662_166209

theorem binary_multiplication_and_shift :
  let a : Nat := 109  -- 1101101₂ in decimal
  let b : Nat := 15   -- 1111₂ in decimal
  let product : Nat := a * b
  let shifted : Rat := (product : Rat) / 4  -- Shifting 2 places right is equivalent to dividing by 4
  shifted = 1010011111.25 := by sorry

end binary_multiplication_and_shift_l1662_166209


namespace money_ratio_l1662_166244

/-- Prove that the ratio of Alison's money to Brittany's money is 1:2 -/
theorem money_ratio (kent_money : ℝ) (brooke_money : ℝ) (brittany_money : ℝ) (alison_money : ℝ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : brittany_money = 4 * brooke_money)
  (h4 : alison_money = 4000) :
  alison_money / brittany_money = 1 / 2 := by
sorry

end money_ratio_l1662_166244


namespace composite_quotient_l1662_166265

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_quotient :
  (product_list first_eight_composites) / (product_list next_eight_composites) = 1 / 1430 := by
  sorry

end composite_quotient_l1662_166265


namespace usable_area_formula_l1662_166238

/-- The usable area of a rectangular field with flooded region -/
def usableArea (x : ℝ) : ℝ :=
  (x + 9) * (x + 7) - (2 * x - 2) * (x - 1)

/-- Theorem stating the usable area of the field -/
theorem usable_area_formula (x : ℝ) : 
  usableArea x = -x^2 + 20*x + 61 := by
  sorry

end usable_area_formula_l1662_166238


namespace kennel_dogs_l1662_166268

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end kennel_dogs_l1662_166268


namespace x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1662_166216

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l1662_166216


namespace line_segment_point_sum_l1662_166203

/-- Given a line y = -2/3x + 6, prove that the sum of coordinates of point T
    satisfies r + s = 8.25, where T(r,s) is on PQ, P and Q are x and y intercepts,
    and area of POQ is 4 times area of TOP. -/
theorem line_segment_point_sum (x₁ y₁ r s : ℝ) : 
  y₁ = 6 ∧                        -- Q is (0, y₁)
  x₁ = 9 ∧                        -- P is (x₁, 0)
  s = -2/3 * r + 6 ∧              -- T(r,s) is on the line
  0 ≤ r ∧ r ≤ x₁ ∧                -- T is between P and Q
  1/2 * x₁ * y₁ = 4 * (1/2 * r * s) -- Area POQ = 4 * Area TOP
  → r + s = 8.25 := by
    sorry

end line_segment_point_sum_l1662_166203


namespace hyperbola_properties_l1662_166267

-- Define the hyperbola
def Hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

-- Define the asymptotes
def Asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def Asymptote2 (x y : ℝ) : Prop := y = -3 * x - 2

theorem hyperbola_properties :
  ∃ (a b h k : ℝ),
    (∀ x y, Asymptote1 x y ∨ Asymptote2 x y → Hyperbola a b h k x y) ∧
    Hyperbola a b h k 1 9 ∧
    a + h = (21 * Real.sqrt 6 - 8) / 6 :=
by sorry

end hyperbola_properties_l1662_166267


namespace certain_number_l1662_166271

theorem certain_number : ∃ x : ℤ, x - 9 = 5 ∧ x = 14 := by sorry

end certain_number_l1662_166271


namespace may_savings_l1662_166285

def savings (month : ℕ) : ℕ :=
  match month with
  | 0 => 20  -- January
  | 1 => 3 * 20  -- February
  | n + 2 => 3 * savings (n + 1) + 50  -- March onwards

theorem may_savings : savings 4 = 2270 := by
  sorry

end may_savings_l1662_166285


namespace gcd_from_lcm_and_ratio_l1662_166288

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  gcd A B = 30 := by
sorry

end gcd_from_lcm_and_ratio_l1662_166288


namespace intersection_of_A_and_B_l1662_166246

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1662_166246


namespace cordelia_bleaching_time_l1662_166264

/-- Represents the time in hours for a hair coloring process -/
structure HairColoringTime where
  bleaching : ℝ
  dyeing : ℝ

/-- The properties of Cordelia's hair coloring process -/
def cordelias_hair_coloring (t : HairColoringTime) : Prop :=
  t.bleaching + t.dyeing = 9 ∧ t.dyeing = 2 * t.bleaching

theorem cordelia_bleaching_time :
  ∀ t : HairColoringTime, cordelias_hair_coloring t → t.bleaching = 3 := by
  sorry

end cordelia_bleaching_time_l1662_166264


namespace baseball_card_price_l1662_166249

/-- Given the following conditions:
  - 2 packs of basketball cards were bought at $3 each
  - 5 decks of baseball cards were bought
  - A $50 bill was used for payment
  - $24 was received in change
  Prove that the price of each baseball card deck is $4 -/
theorem baseball_card_price 
  (basketball_packs : ℕ)
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (total_paid : ℕ)
  (change_received : ℕ)
  (h1 : basketball_packs = 2)
  (h2 : basketball_price = 3)
  (h3 : baseball_decks = 5)
  (h4 : total_paid = 50)
  (h5 : change_received = 24) :
  (total_paid - change_received - basketball_packs * basketball_price) / baseball_decks = 4 :=
by sorry

end baseball_card_price_l1662_166249


namespace reflected_ray_equation_l1662_166286

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The incident ray passes through these points -/
def M : Point := { x := 3, y := -2 }
def P : Point := { x := 0, y := 1 }

/-- P is on the y-axis -/
axiom P_on_y_axis : P.x = 0

/-- Function to check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

/-- The reflected ray -/
def reflected_ray : Line := { A := 1, B := -1, C := 1 }

/-- Theorem stating that the reflected ray has the equation x - y + 1 = 0 -/
theorem reflected_ray_equation :
  point_on_line P reflected_ray ∧
  point_on_line { x := -M.x, y := M.y } reflected_ray :=
sorry

end reflected_ray_equation_l1662_166286


namespace min_value_sum_reciprocals_l1662_166273

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) ≥ 3 ∧
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end min_value_sum_reciprocals_l1662_166273


namespace arithmetic_sequence_first_term_l1662_166287

/-- An arithmetic sequence of integers -/
def ArithSeq (a₁ d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => ArithSeq a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def ArithSeqSum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_first_term (a₁ : ℤ) :
  (∃ d : ℤ, d > 0 ∧
    let S := ArithSeqSum a₁ d 9
    (ArithSeq a₁ d 4) * (ArithSeq a₁ d 17) > S - 4 ∧
    (ArithSeq a₁ d 12) * (ArithSeq a₁ d 9) < S + 60) →
  a₁ ∈ ({-10, -9, -8, -7, -5, -4, -3, -2} : Set ℤ) := by
  sorry

end arithmetic_sequence_first_term_l1662_166287


namespace even_sum_digits_all_residues_l1662_166254

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Main theorem -/
theorem even_sum_digits_all_residues (k : ℕ) (h : k ≥ 2) :
  ∀ r, r < k → ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r :=
by sorry

end even_sum_digits_all_residues_l1662_166254


namespace larger_number_proof_l1662_166219

/-- Given two positive integers with HCF 23 and LCM factors 13 and 16, prove the larger number is 368 -/
theorem larger_number_proof (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 23 → Nat.lcm a b = 23 * 13 * 16 → max a b = 368 := by
  sorry

end larger_number_proof_l1662_166219


namespace rectangle_sides_from_ratio_and_area_l1662_166262

theorem rectangle_sides_from_ratio_and_area 
  (m n S : ℝ) (hm : m > 0) (hn : n > 0) (hS : S > 0) :
  ∃ (x y : ℝ), 
    x / y = m / n ∧ 
    x * y = S ∧ 
    x = Real.sqrt ((m * S) / n) ∧ 
    y = Real.sqrt ((n * S) / m) :=
by sorry

end rectangle_sides_from_ratio_and_area_l1662_166262


namespace octal_subtraction_example_l1662_166230

/-- Subtraction in octal (base 8) number system --/
def octal_subtraction (a b : Nat) : Nat :=
  -- Define octal subtraction here
  sorry

/-- Conversion from decimal to octal --/
def decimal_to_octal (n : Nat) : Nat :=
  -- Define decimal to octal conversion here
  sorry

/-- Conversion from octal to decimal --/
def octal_to_decimal (n : Nat) : Nat :=
  -- Define octal to decimal conversion here
  sorry

theorem octal_subtraction_example : octal_subtraction 325 237 = 66 := by
  sorry

end octal_subtraction_example_l1662_166230


namespace calculate_expression_l1662_166228

theorem calculate_expression : -1^2023 + 8 / (-2)^2 - |-4| * 5 = -19 := by
  sorry

end calculate_expression_l1662_166228


namespace equation_solutions_l1662_166270

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (Real.sqrt x) = 15 / (8 - Real.sqrt (Real.sqrt x))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 625 ∨ x = 81) :=
by sorry

end equation_solutions_l1662_166270


namespace sale_price_percentage_l1662_166239

theorem sale_price_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := original_price * 0.5
  let final_price := first_sale_price * 0.9
  (final_price / original_price) * 100 = 45 := by
  sorry

end sale_price_percentage_l1662_166239


namespace quadratic_inequality_implies_zero_l1662_166208

theorem quadratic_inequality_implies_zero (a b x y : ℤ) 
  (h1 : a > b^2) 
  (h2 : a^2 * x^2 + 2*a*b * x*y + (b^2 + 1) * y^2 < b^2 + 1) : 
  x = 0 ∧ y = 0 := by
  sorry

end quadratic_inequality_implies_zero_l1662_166208


namespace fraction_of_25_l1662_166275

theorem fraction_of_25 : ∃ x : ℚ, x * 25 = 0.9 * 40 - 16 ∧ x = 4 / 5 := by
  sorry

end fraction_of_25_l1662_166275


namespace box_volume_problem_l1662_166281

theorem box_volume_problem :
  ∃! (x : ℕ), 
    x > 3 ∧ 
    (x + 3) * (x - 3) * (x^2 + 9) < 500 := by
  sorry

end box_volume_problem_l1662_166281


namespace inscribed_square_area_l1662_166298

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

-- Define the square
structure InscribedSquare where
  center : ℝ
  side_half : ℝ

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : InscribedSquare),
    s.center = 5 ∧
    parabola (s.center + s.side_half) = -2 * s.side_half ∧
    (2 * s.side_half)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end inscribed_square_area_l1662_166298


namespace rectangle_count_num_rectangles_nat_l1662_166259

/-- The number of rectangles formed in a rectangle ABCD with additional points and lines -/
theorem rectangle_count (m n : ℕ) : 
  (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4 = 
  (Nat.choose (m + 2) 2) * (Nat.choose (n + 2) 2) :=
by sorry

/-- The formula for the number of rectangles formed -/
def num_rectangles (m n : ℕ) : ℕ := (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4

/-- The number of rectangles is always a natural number -/
theorem num_rectangles_nat (m n : ℕ) : 
  ∃ k : ℕ, num_rectangles m n = k :=
by sorry

end rectangle_count_num_rectangles_nat_l1662_166259


namespace number_of_students_l1662_166290

theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_average : ℝ) :
  initial_average = 100 →
  wrong_mark = 90 →
  correct_mark = 10 →
  correct_average = 92 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * initial_average - (wrong_mark - correct_mark) = (n : ℝ) * correct_average ∧ n = 10 :=
by sorry

end number_of_students_l1662_166290


namespace solution_set_equivalence_l1662_166242

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the given inequality
def givenInequality (k a b c x : ℝ) : Prop :=
  (k / (x + a) + (x + b) / (x + c)) < 0

-- Define the target inequality
def targetInequality (k a b c x : ℝ) : Prop :=
  (k * x / (a * x + 1) + (b * x + 1) / (c * x + 1)) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (h : SolutionSet = {x | x ∈ (Set.Ioo (-1) (-1/3) ∪ Set.Ioo (1/2) 1) ∧ givenInequality k a b c x}) :
  SolutionSet = {x | x ∈ (Set.Ioo (-3) (-1) ∪ Set.Ioo 1 2) ∧ targetInequality k a b c x} :=
by sorry

end solution_set_equivalence_l1662_166242


namespace systematic_sampling_first_two_samples_l1662_166247

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sample : Nat

/-- Calculates the interval between samples -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the first sampled number -/
def first_sample (s : SystematicSampling) : Nat :=
  s.last_sample % (sample_interval s)

/-- Calculates the second sampled number -/
def second_sample (s : SystematicSampling) : Nat :=
  first_sample s + sample_interval s

/-- Theorem stating the first two sampled numbers for the given scenario -/
theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sample = 7900) :
  first_sample s = 60 ∧ second_sample s = 220 := by
  sorry


end systematic_sampling_first_two_samples_l1662_166247


namespace num_parallelepipeds_is_29_l1662_166200

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in 3D space -/
def FourPoints := Fin 4 → Point3D

/-- Predicate to check if four points are non-coplanar -/
def NonCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    ∀ (i : Fin 4), a * (points i).x + b * (points i).y + c * (points i).z + d = 0

/-- The number of distinct parallelepipeds that can be formed -/
def NumParallelepipeds (points : FourPoints) : ℕ := 29

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem num_parallelepipeds_is_29 (points : FourPoints) (h : NonCoplanar points) :
  NumParallelepipeds points = 29 := by
  sorry

end num_parallelepipeds_is_29_l1662_166200


namespace sue_grocery_spending_l1662_166233

/-- Calculates Sue's spending on a grocery shopping trip with specific conditions --/
theorem sue_grocery_spending : 
  let apple_price : ℚ := 2
  let apple_quantity : ℕ := 4
  let juice_price : ℚ := 6
  let juice_quantity : ℕ := 2
  let bread_price : ℚ := 3
  let bread_quantity : ℕ := 3
  let cheese_price : ℚ := 4
  let cheese_quantity : ℕ := 2
  let cereal_price : ℚ := 8
  let cereal_quantity : ℕ := 1
  let cheese_discount : ℚ := 0.25
  let order_discount_threshold : ℚ := 40
  let order_discount_rate : ℚ := 0.1

  let discounted_cheese_price : ℚ := cheese_price * (1 - cheese_discount)
  let subtotal : ℚ := 
    apple_price * apple_quantity +
    juice_price * juice_quantity +
    bread_price * bread_quantity +
    discounted_cheese_price * cheese_quantity +
    cereal_price * cereal_quantity

  let final_total : ℚ := 
    if subtotal ≥ order_discount_threshold
    then subtotal * (1 - order_discount_rate)
    else subtotal

  final_total = 387/10 := by sorry

end sue_grocery_spending_l1662_166233


namespace chess_team_boys_count_l1662_166206

theorem chess_team_boys_count 
  (total_members : ℕ) 
  (meeting_attendance : ℕ) 
  (h1 : total_members = 26)
  (h2 : meeting_attendance = 16)
  : ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + girls / 2 = meeting_attendance ∧
    boys = 6 := by
  sorry

end chess_team_boys_count_l1662_166206


namespace expression_simplification_l1662_166253

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : 3 * x + y / 3 + 2 * z ≠ 0) :
  (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹) = 
  (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z)) :=
by sorry

end expression_simplification_l1662_166253


namespace dhoni_leftover_earnings_l1662_166213

/-- Calculates the percentage of earnings left over after Dhoni's expenses --/
theorem dhoni_leftover_earnings (rent : ℝ) (utilities : ℝ) (groceries : ℝ) (transportation : ℝ)
  (h_rent : rent = 25)
  (h_utilities : utilities = 15)
  (h_groceries : groceries = 20)
  (h_transportation : transportation = 12) :
  100 - (rent + (rent - rent * 0.1) + utilities + groceries + transportation) = 5.5 := by
  sorry

end dhoni_leftover_earnings_l1662_166213


namespace quadratic_roots_value_l1662_166269

theorem quadratic_roots_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 49/5 := by
sorry

end quadratic_roots_value_l1662_166269


namespace robe_cost_is_two_l1662_166210

/-- Calculates the cost per robe given the total number of singers, existing robes, and total cost for new robes. -/
def cost_per_robe (total_singers : ℕ) (existing_robes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_singers - existing_robes)

/-- Proves that the cost per robe is $2 given the specific conditions of the problem. -/
theorem robe_cost_is_two :
  cost_per_robe 30 12 36 = 2 := by
  sorry

end robe_cost_is_two_l1662_166210


namespace min_value_theorem_l1662_166258

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + 3*d = 5*c*d ∧ 3*c + 4*d = 5 :=
by sorry

end min_value_theorem_l1662_166258


namespace alternating_sequence_solution_l1662_166295

theorem alternating_sequence_solution (n : ℕ) (h : n ≥ 4) :
  ∃! (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ 2*n → a i > 0) ∧
    (∀ k, 0 ≤ k ∧ k < n →
      a (2*k+1) = 1/(a (2*n)) + 1/(a (2*k+2)) ∧
      a (2*k+2) = a (2*k+1) + a (2*k+3)) ∧
    (a (2*n) = a (2*n-1) + a 1) →
  ∀ k, 0 ≤ k ∧ k < n → a (2*k+1) = 1 ∧ a (2*k+2) = 2 :=
by sorry

end alternating_sequence_solution_l1662_166295


namespace birds_on_fence_l1662_166240

theorem birds_on_fence (initial_storks : ℕ) (additional_storks : ℕ) (total_after : ℕ) 
  (h1 : initial_storks = 4)
  (h2 : additional_storks = 6)
  (h3 : total_after = 13) :
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_storks = total_after ∧ initial_birds = 3 := by
  sorry

end birds_on_fence_l1662_166240


namespace triangle_side_sum_unbounded_l1662_166243

theorem triangle_side_sum_unbounded (b c : ℝ) :
  ∀ ε > 0, ∃ b' c' : ℝ,
    b' > 0 ∧ c' > 0 ∧
    b'^2 + c'^2 + b' * c' = 25 ∧
    b' + c' > ε :=
by sorry

end triangle_side_sum_unbounded_l1662_166243


namespace square_expansion_l1662_166255

theorem square_expansion (n : ℕ) (h : ∃ k : ℕ, k > 0 ∧ (n + k)^2 - n^2 = 47) : n = 23 := by
  sorry

end square_expansion_l1662_166255


namespace shaded_area_calculation_l1662_166294

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : (Circle × Circle × Circle) :=
  let small_circle : Circle := { center := (0, 0), radius := 2 }
  let large_circle1 : Circle := { center := (-2, 0), radius := 3 }
  let large_circle2 : Circle := { center := (2, 0), radius := 3 }
  (small_circle, large_circle1, large_circle2)

-- Define the shaded area function
noncomputable def shaded_area (setup : Circle × Circle × Circle) : ℝ :=
  2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area problem_setup = 2 * Real.pi - 4 * Real.sqrt 5 := by
  sorry

end shaded_area_calculation_l1662_166294


namespace dispersion_measures_l1662_166205

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define statistics
def standardDeviation (s : Sample) : Real :=
  sorry

def median (s : Sample) : Real :=
  sorry

def range (s : Sample) : Real :=
  sorry

def mean (s : Sample) : Real :=
  sorry

-- Define a predicate for measures of dispersion
def measuresDispersion (f : Sample → Real) : Prop :=
  sorry

-- Theorem statement
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (median) ∧
  ¬measuresDispersion (mean) :=
sorry

end dispersion_measures_l1662_166205


namespace dore_change_l1662_166235

/-- The amount of change Mr. Doré receives after his purchase -/
def change (pants_cost shirt_cost tie_cost payment : ℕ) : ℕ :=
  payment - (pants_cost + shirt_cost + tie_cost)

/-- Theorem stating that Mr. Doré receives $2 in change -/
theorem dore_change : change 140 43 15 200 = 2 := by
  sorry

end dore_change_l1662_166235


namespace interior_angles_sum_l1662_166283

/-- Given a convex polygon where the sum of interior angles is 3600 degrees,
    prove that the sum of interior angles of a polygon with 3 more sides is 4140 degrees. -/
theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3600) → (180 * ((n + 3) - 2) = 4140) := by
  sorry

end interior_angles_sum_l1662_166283


namespace integer_solution_existence_l1662_166234

theorem integer_solution_existence (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end integer_solution_existence_l1662_166234


namespace arctan_sum_three_four_l1662_166229

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end arctan_sum_three_four_l1662_166229


namespace area_of_rectangle_PQRS_l1662_166231

-- Define the Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the Rectangle type
structure Rectangle :=
  (p : Point) (q : Point) (r : Point) (s : Point)

-- Define the area function for a rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.q.x - rect.p.x)
  let height := abs (rect.p.y - rect.s.y)
  width * height

-- Theorem statement
theorem area_of_rectangle_PQRS :
  let p := Point.mk (-4) 2
  let q := Point.mk 4 2
  let r := Point.mk 4 (-2)
  let s := Point.mk (-4) (-2)
  let rect := Rectangle.mk p q r s
  rectangleArea rect = 32 := by
  sorry

end area_of_rectangle_PQRS_l1662_166231


namespace jogs_five_miles_per_day_l1662_166282

/-- Represents the number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Represents the number of weeks -/
def num_weeks : ℕ := 3

/-- Represents the total miles run over the given weeks -/
def total_miles : ℕ := 75

/-- Calculates the number of miles jogged per day -/
def miles_per_day : ℚ :=
  total_miles / (weekdays_per_week * num_weeks)

/-- Theorem stating that the person jogs 5 miles per day -/
theorem jogs_five_miles_per_day : miles_per_day = 5 := by
  sorry

end jogs_five_miles_per_day_l1662_166282


namespace min_value_sum_of_reciprocals_l1662_166280

theorem min_value_sum_of_reciprocals (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end min_value_sum_of_reciprocals_l1662_166280


namespace rectangle_in_circle_l1662_166291

theorem rectangle_in_circle (d p : ℝ) (h_d_pos : d > 0) (h_p_pos : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≥ y ∧
  (2 * x + 2 * y = p) ∧  -- perimeter condition
  (x^2 + y^2 = d^2) ∧    -- inscribed in circle condition
  (x - y = d) :=
sorry

end rectangle_in_circle_l1662_166291


namespace max_toys_frank_can_buy_l1662_166241

/-- Represents the types of toys available --/
inductive Toy
| SmallCar
| Puzzle
| LegoSet

/-- Returns the price of a toy --/
def toyPrice (t : Toy) : ℕ :=
  match t with
  | Toy.SmallCar => 8
  | Toy.Puzzle => 12
  | Toy.LegoSet => 20

/-- Represents a shopping cart with toys --/
structure Cart :=
  (smallCars : ℕ)
  (puzzles : ℕ)
  (legoSets : ℕ)

/-- Calculates the total cost of a cart, considering the promotion --/
def cartCost (c : Cart) : ℕ :=
  (c.smallCars / 3 * 2 + c.smallCars % 3) * toyPrice Toy.SmallCar +
  (c.puzzles / 3 * 2 + c.puzzles % 3) * toyPrice Toy.Puzzle +
  (c.legoSets / 3 * 2 + c.legoSets % 3) * toyPrice Toy.LegoSet

/-- Calculates the total number of toys in a cart --/
def cartSize (c : Cart) : ℕ :=
  c.smallCars + c.puzzles + c.legoSets

/-- Theorem: The maximum number of toys Frank can buy with $40 is 6 --/
theorem max_toys_frank_can_buy :
  ∀ c : Cart, cartCost c ≤ 40 → cartSize c ≤ 6 :=
sorry

end max_toys_frank_can_buy_l1662_166241


namespace max_drumming_bunnies_l1662_166227

/-- Represents a drum with a specific size -/
structure Drum where
  size : ℕ

/-- Represents a pair of drumsticks with a specific length -/
structure Drumsticks where
  length : ℕ

/-- Represents a bunny with its assigned drum and drumsticks -/
structure Bunny where
  drum : Drum
  sticks : Drumsticks

/-- Determines if a bunny can drum based on its drum and sticks compared to another bunny -/
def canDrum (b1 b2 : Bunny) : Prop :=
  b1.drum.size > b2.drum.size ∧ b1.sticks.length > b2.sticks.length

theorem max_drumming_bunnies 
  (bunnies : Fin 7 → Bunny)
  (h_diff_drums : ∀ i j, i ≠ j → (bunnies i).drum.size ≠ (bunnies j).drum.size)
  (h_diff_sticks : ∀ i j, i ≠ j → (bunnies i).sticks.length ≠ (bunnies j).sticks.length) :
  ∃ (drummers : Finset (Fin 7)),
    drummers.card = 6 ∧
    ∀ i ∈ drummers, ∃ j, canDrum (bunnies i) (bunnies j) :=
by
  sorry

end max_drumming_bunnies_l1662_166227


namespace circle_ratio_l1662_166251

theorem circle_ratio (r R a c : Real) (hr : r > 0) (hR : R > r) (ha : a > c) (hc : c > 0) :
  π * R^2 = (a - c) * (π * R^2 - π * r^2) →
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) := by
  sorry

end circle_ratio_l1662_166251


namespace line_l_is_correct_l1662_166252

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

-- Theorem statement
theorem line_l_is_correct :
  (∀ x y : ℝ, line_l x y → (x, y) = point_A ∨ (x, y) ≠ point_A) ∧
  (∀ x y : ℝ, line_l x y → given_line x y → False) ∧
  line_l point_A.1 point_A.2 :=
sorry

end line_l_is_correct_l1662_166252


namespace cuboid_volume_l1662_166226

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) :
  a * b * c = 6 := by
  sorry

end cuboid_volume_l1662_166226


namespace polygon_triangulation_l1662_166277

theorem polygon_triangulation (n : ℕ) :
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (n - 2 = 7) →  -- Number of triangles formed is n - 2, which equals 7
  n = 9 := by
sorry

end polygon_triangulation_l1662_166277


namespace defective_pens_l1662_166236

theorem defective_pens (total : ℕ) (prob : ℚ) (defective : ℕ) : 
  total = 8 →
  prob = 15/28 →
  (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1) = prob →
  defective = 2 :=
sorry

end defective_pens_l1662_166236


namespace dice_product_divisible_by_8_l1662_166284

/-- The number of dice rolled simultaneously -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability that a single die roll is divisible by 2 -/
def prob_divisible_by_2 : ℚ := 1/2

/-- The probability that the product of dice rolls is divisible by 8 -/
def prob_product_divisible_by_8 : ℚ := 247/256

/-- Theorem: The probability that the product of 8 standard 6-sided dice rolls is divisible by 8 is 247/256 -/
theorem dice_product_divisible_by_8 :
  prob_product_divisible_by_8 = 247/256 :=
sorry

end dice_product_divisible_by_8_l1662_166284


namespace minimum_fraction_ponies_with_horseshoes_l1662_166297

theorem minimum_fraction_ponies_with_horseshoes :
  ∀ (num_ponies num_horses num_ponies_with_horseshoes num_icelandic_ponies_with_horseshoes : ℕ),
  num_horses = num_ponies + 4 →
  num_horses + num_ponies ≥ 164 →
  8 * num_icelandic_ponies_with_horseshoes = 5 * num_ponies_with_horseshoes →
  num_ponies_with_horseshoes ≤ num_ponies →
  (∃ (min_fraction : ℚ), 
    min_fraction = num_ponies_with_horseshoes / num_ponies ∧
    min_fraction = 1 / 10) :=
by sorry

end minimum_fraction_ponies_with_horseshoes_l1662_166297


namespace number_of_selection_schemes_l1662_166256

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The total number of teachers -/
def total_teachers : ℕ := num_male + num_female

/-- The number of teachers to be selected -/
def teachers_to_select : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid selection schemes -/
theorem number_of_selection_schemes : 
  permutations total_teachers teachers_to_select - 
  (permutations num_male teachers_to_select + 
   permutations num_female teachers_to_select) = 420 := by
  sorry

end number_of_selection_schemes_l1662_166256


namespace five_thursdays_in_august_l1662_166202

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its dates -/
structure Month :=
  (dates : List Date)
  (numDays : Nat)

def july : Month := sorry
def august : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem five_thursdays_in_august 
  (h1 : july.numDays = 31)
  (h2 : august.numDays = 31)
  (h3 : countDaysInMonth july DayOfWeek.Tuesday = 5) :
  countDaysInMonth august DayOfWeek.Thursday = 5 := by sorry

end five_thursdays_in_august_l1662_166202


namespace probability_at_least_two_black_l1662_166296

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least 2 black balls when drawing 3 balls from a bag 
    containing 5 white balls and 3 black balls -/
theorem probability_at_least_two_black : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 + Nat.choose black_balls 3) / 
  Nat.choose total_balls drawn_balls = 2 / 7 := by
  sorry

end probability_at_least_two_black_l1662_166296


namespace line_intersects_parabola_vertex_once_l1662_166225

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^3 - 3ax + a^2 is exactly one. -/
theorem line_intersects_parabola_vertex_once :
  ∃! a : ℝ, ∃ x y : ℝ,
    (y = x + a) ∧                   -- Line equation
    (y = x^3 - 3*a*x + a^2) ∧       -- Parabola equation
    (∀ x' : ℝ, x'^3 - 3*a*x' + a^2 ≤ x^3 - 3*a*x + a^2) -- Vertex condition
    := by sorry

end line_intersects_parabola_vertex_once_l1662_166225


namespace tree_height_after_four_months_l1662_166248

/-- Calculates the height of a tree after a given number of months -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (growth_period : ℕ) (months : ℕ) : ℕ :=
  initial_height * 100 + (months * 4 / growth_period) * growth_rate

/-- Theorem stating that a tree with given growth parameters reaches 600 cm after 4 months -/
theorem tree_height_after_four_months :
  tree_height 2 50 2 4 = 600 := by
  sorry

#eval tree_height 2 50 2 4

end tree_height_after_four_months_l1662_166248


namespace unique_function_satisfying_equation_l1662_166207

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - 4 * x^2 * f y :=
by
  sorry

end unique_function_satisfying_equation_l1662_166207


namespace james_pays_37_50_l1662_166237

/-- Calculates the amount James pays for singing lessons given the specified conditions. -/
def james_payment (total_lessons : ℕ) (lesson_cost : ℚ) (free_lessons : ℕ) (fully_paid_lessons : ℕ) (uncle_contribution : ℚ) : ℚ :=
  let remaining_lessons := total_lessons - free_lessons - fully_paid_lessons
  let partially_paid_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := fully_paid_lessons + partially_paid_lessons
  let total_cost := total_paid_lessons * lesson_cost
  total_cost * (1 - uncle_contribution)

/-- Theorem stating that James pays $37.50 for his singing lessons. -/
theorem james_pays_37_50 :
  james_payment 20 5 1 10 (1/2) = 37.5 := by
  sorry

end james_pays_37_50_l1662_166237


namespace max_sum_at_vertex_l1662_166266

/-- Represents a face of the cube -/
structure Face :=
  (number : ℕ)

/-- Represents a cube with six numbered faces -/
structure Cube :=
  (faces : Fin 6 → Face)
  (opposite_sum : ∀ i : Fin 3, (faces i).number + (faces (i + 3)).number = 10)

/-- Represents a vertex of the cube -/
structure Vertex :=
  (face1 : Face)
  (face2 : Face)
  (face3 : Face)

/-- The theorem stating the maximum sum at a vertex -/
theorem max_sum_at_vertex (c : Cube) : 
  (∃ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                 v.face2 ∈ Set.range c.faces ∧ 
                 v.face3 ∈ Set.range c.faces ∧ 
                 v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3) →
  (∀ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                v.face2 ∈ Set.range c.faces ∧ 
                v.face3 ∈ Set.range c.faces ∧ 
                v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3 →
                v.face1.number + v.face2.number + v.face3.number ≤ 22) :=
sorry

end max_sum_at_vertex_l1662_166266


namespace art_collection_cost_l1662_166289

/-- The total cost of John's art collection --/
def total_cost (first_3_price : ℚ) : ℚ :=
  -- Cost of first 3 pieces
  3 * first_3_price +
  -- Cost of next 2 pieces (25% more expensive)
  2 * (first_3_price * (1 + 1/4)) +
  -- Cost of last 3 pieces (50% more expensive)
  3 * (first_3_price * (1 + 1/2))

/-- Theorem stating the total cost of John's art collection --/
theorem art_collection_cost :
  ∃ (first_3_price : ℚ),
    first_3_price > 0 ∧
    3 * first_3_price = 45000 ∧
    total_cost first_3_price = 150000 := by
  sorry


end art_collection_cost_l1662_166289


namespace apple_juice_problem_l1662_166201

theorem apple_juice_problem (x y : ℝ) : 
  (x - 1 = y + 1) →  -- Equalizing condition
  (x + 9 = 30) →     -- First barrel full after transfer
  (y - 9 = 10) →     -- Second barrel one-third full after transfer
  (x = 21 ∧ y = 19 ∧ x + y = 40) := by
  sorry

end apple_juice_problem_l1662_166201


namespace sqrt_product_equals_product_l1662_166217

theorem sqrt_product_equals_product : Real.sqrt (9 * 16) = 3 * 4 := by
  sorry

end sqrt_product_equals_product_l1662_166217


namespace arithmetic_sequence_ratio_l1662_166232

/-- Given two arithmetic sequences, this theorem proves that if the ratio of their sums
    follows a specific pattern, then the ratio of their 7th terms is 13/20. -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence a
  (∀ n, T n = (n * (b 1 + b n)) / 2) →  -- Sum formula for arithmetic sequence b
  (∀ n, S n / T n = n / (n + 7)) →      -- Given condition
  a 7 / b 7 = 13 / 20 :=
by sorry

end arithmetic_sequence_ratio_l1662_166232


namespace right_triangle_sin_c_l1662_166222

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi)
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 7 / 25) :
  Real.sin C = 24 / 25 := by
  sorry

end right_triangle_sin_c_l1662_166222


namespace smaller_field_area_l1662_166293

theorem smaller_field_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 10 →
  smaller_area = 225 := by
sorry

end smaller_field_area_l1662_166293


namespace greatest_product_three_digit_l1662_166212

def Digits : Finset Nat := {3, 5, 7, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit (d e : Nat) : Nat := 10 * d + e

def one_odd_one_even (x y : Nat) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)

theorem greatest_product_three_digit :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    one_odd_one_even (three_digit a b c) (two_digit d e) →
    three_digit a b c * two_digit d e ≤ 972 * 85 :=
  sorry

end greatest_product_three_digit_l1662_166212


namespace min_value_theorem_l1662_166263

theorem min_value_theorem (x y : ℝ) 
  (h : ∀ (n : ℕ), n > 0 → n * x + (1 / n) * y ≥ 1) :
  (∀ (a b : ℝ), (∀ (n : ℕ), n > 0 → n * a + (1 / n) * b ≥ 1) → 41 * x + 2 * y ≤ 41 * a + 2 * b) ∧ 
  (∃ (x₀ y₀ : ℝ), (∀ (n : ℕ), n > 0 → n * x₀ + (1 / n) * y₀ ≥ 1) ∧ 41 * x₀ + 2 * y₀ = 9) :=
by sorry

end min_value_theorem_l1662_166263


namespace amit_work_days_l1662_166221

theorem amit_work_days (ananthu_days : ℕ) (amit_worked : ℕ) (total_days : ℕ) :
  ananthu_days = 30 →
  amit_worked = 3 →
  total_days = 27 →
  ∃ (amit_days : ℕ),
    amit_days = 15 ∧
    (3 : ℝ) / amit_days + (total_days - amit_worked : ℝ) / ananthu_days = 1 :=
by sorry

end amit_work_days_l1662_166221


namespace new_supervisor_salary_l1662_166299

def problem (workers : ℕ) (supervisors : ℕ) (initial_avg : ℝ) (supervisor_a : ℝ) (supervisor_b : ℝ) (supervisor_c : ℝ) (new_avg : ℝ) : Prop :=
  let total_people := workers + supervisors
  let initial_total := initial_avg * total_people
  let workers_supervisors_ab_total := initial_total - supervisor_c
  let new_total := new_avg * total_people
  let salary_difference := initial_total - new_total
  let new_supervisor_salary := supervisor_c - salary_difference
  new_supervisor_salary = 4600

theorem new_supervisor_salary :
  problem 15 3 5300 6200 7200 8200 5100 :=
by
  sorry

end new_supervisor_salary_l1662_166299


namespace imaginary_part_of_z_l1662_166272

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) : 
  z.im = (1 - Real.sqrt 2) / 2 := by
sorry

end imaginary_part_of_z_l1662_166272


namespace max_value_theorem_l1662_166204

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  2*a*b + 2*b*c*Real.sqrt 2 + 2*a*c ≤ 2*(1 + Real.sqrt 2)/3 := by
  sorry

end max_value_theorem_l1662_166204


namespace sin_squared_value_l1662_166261

theorem sin_squared_value (α : Real) (h : Real.tan (α + π/4) = 3/4) :
  Real.sin (π/4 - α) ^ 2 = 16/25 := by sorry

end sin_squared_value_l1662_166261


namespace vegetarian_eaters_count_l1662_166276

/-- Represents the eating habits in a family -/
structure FamilyDiet where
  onlyVegetarian : ℕ
  onlyNonVegetarian : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def vegetarianEaters (f : FamilyDiet) : ℕ :=
  f.onlyVegetarian + f.both

/-- Theorem: Given the family diet information, prove that the number of vegetarian eaters
    is the sum of those who eat only vegetarian and those who eat both -/
theorem vegetarian_eaters_count (f : FamilyDiet) 
    (h1 : f.onlyVegetarian = 13)
    (h2 : f.onlyNonVegetarian = 7)
    (h3 : f.both = 8) :
    vegetarianEaters f = 21 := by
  sorry

end vegetarian_eaters_count_l1662_166276


namespace min_four_dollar_frisbees_l1662_166250

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 196 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      (∃ (other_three_dollar : ℕ),
        other_three_dollar + other_four_dollar = total_frisbees ∧
        3 * other_three_dollar + 4 * other_four_dollar = total_receipts) →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 4 :=
by sorry

end min_four_dollar_frisbees_l1662_166250


namespace square_of_difference_l1662_166215

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end square_of_difference_l1662_166215


namespace square_area_from_adjacent_vertices_l1662_166220

/-- Given two points (1,6) and (5,2) as adjacent vertices of a square, prove that the area of the square is 32. -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, 6)
  let p2 : ℝ × ℝ := (5, 2)
  32 = (((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) : ℝ) := by sorry

end square_area_from_adjacent_vertices_l1662_166220


namespace martin_goldfish_purchase_l1662_166279

/-- The number of new goldfish Martin purchases every week -/
def new_goldfish_per_week : ℕ := sorry

theorem martin_goldfish_purchase :
  let initial_goldfish : ℕ := 18
  let dying_goldfish_per_week : ℕ := 5
  let weeks : ℕ := 7
  let final_goldfish : ℕ := 4
  final_goldfish = initial_goldfish + (new_goldfish_per_week - dying_goldfish_per_week) * weeks →
  new_goldfish_per_week = 3 := by
  sorry

end martin_goldfish_purchase_l1662_166279
