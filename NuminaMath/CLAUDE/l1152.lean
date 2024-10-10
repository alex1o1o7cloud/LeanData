import Mathlib

namespace right_triangle_perimeter_l1152_115226

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem right_triangle_perimeter : ∃ (a b c : ℕ),
  a = 11 ∧
  is_pythagorean_triple a b c ∧
  a + b + c = 132 :=
by
  sorry

end right_triangle_perimeter_l1152_115226


namespace direction_vector_b_value_l1152_115263

/-- Given a line passing through two points, prove that its direction vector
    in the form (b, -1) has b = 1. -/
theorem direction_vector_b_value 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-3, 2)) 
  (h2 : p2 = (2, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) : 
  b = 1 := by
sorry

end direction_vector_b_value_l1152_115263


namespace units_digit_of_expression_l1152_115293

theorem units_digit_of_expression : 
  ((5 * 21 * 1933) + 5^4 - (6 * 2 * 1944)) % 10 = 2 := by sorry

end units_digit_of_expression_l1152_115293


namespace function_g_property_l1152_115239

theorem function_g_property (g : ℝ → ℝ) 
  (h1 : ∀ (b c : ℝ), c^2 * g b = b^2 * g c) 
  (h2 : g 3 ≠ 0) : 
  (g 6 - g 4) / g 3 = 20 / 9 := by
  sorry

end function_g_property_l1152_115239


namespace u_1990_equals_one_l1152_115220

def u : ℕ → ℕ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then 1 - u (n / 2) else u (n / 2)

theorem u_1990_equals_one : u 1990 = 1 := by
  sorry

end u_1990_equals_one_l1152_115220


namespace jackpot_probability_correct_l1152_115274

/-- The total number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers to be chosen in each ticket -/
def numbers_per_ticket : ℕ := 6

/-- The number of tickets bought by the player -/
def tickets_bought : ℕ := 100

/-- The probability of hitting the jackpot with the given number of tickets -/
def jackpot_probability : ℚ :=
  tickets_bought / Nat.choose total_numbers numbers_per_ticket

theorem jackpot_probability_correct :
  jackpot_probability = tickets_bought / Nat.choose total_numbers numbers_per_ticket :=
by sorry

end jackpot_probability_correct_l1152_115274


namespace binary_sum_exp_eq_four_l1152_115261

/-- B(n) is the number of ones in the base two expression for the positive integer n -/
def B (n : ℕ+) : ℕ := sorry

/-- The infinite sum of B(n)/(n(n+1)) for n from 1 to infinity -/
noncomputable def infiniteSum : ℝ := sorry

theorem binary_sum_exp_eq_four :
  Real.exp infiniteSum = 4 := by sorry

end binary_sum_exp_eq_four_l1152_115261


namespace minimum_condition_range_l1152_115259

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains its minimum at x = a, then a < -1 or a > 0 -/
theorem minimum_condition_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_min : IsLocalMin f a) :
  a < -1 ∨ a > 0 := by
sorry

end minimum_condition_range_l1152_115259


namespace largest_n_divisible_by_seven_largest_n_is_49999_l1152_115212

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 50000 →
  (3 * (n - 3)^2 - 4 * n + 28) % 7 = 0 →
  n ≤ 49999 :=
by sorry

theorem largest_n_is_49999 : 
  (3 * (49999 - 3)^2 - 4 * 49999 + 28) % 7 = 0 ∧
  ∀ m : ℕ, m > 49999 → m < 50000 → (3 * (m - 3)^2 - 4 * m + 28) % 7 ≠ 0 :=
by sorry

end largest_n_divisible_by_seven_largest_n_is_49999_l1152_115212


namespace exactly_one_pair_probability_l1152_115257

def number_of_pairs : ℕ := 8
def shoes_drawn : ℕ := 4

def total_outcomes : ℕ := (Nat.choose (2 * number_of_pairs) shoes_drawn)

def favorable_outcomes : ℕ :=
  (Nat.choose number_of_pairs 1) *
  (Nat.choose (number_of_pairs - 1) 2) *
  (Nat.choose 2 1) *
  (Nat.choose 2 1)

theorem exactly_one_pair_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 24 / 65 := by
  sorry

end exactly_one_pair_probability_l1152_115257


namespace cube_side_ratio_l1152_115279

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 4 → a / b = 2 := by
  sorry

end cube_side_ratio_l1152_115279


namespace simplify_expression_l1152_115255

theorem simplify_expression (m n : ℝ) (h : m^2 + 3*m*n = 5) :
  5*m^2 - 3*m*n - (-9*m*n + 3*m^2) = 10 := by
sorry

end simplify_expression_l1152_115255


namespace quadratic_function_max_value_l1152_115227

theorem quadratic_function_max_value (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (∀ m : ℝ, 7 * b + 5 * c ≤ m → m ≥ -6) :=
by sorry

end quadratic_function_max_value_l1152_115227


namespace truck_speed_on_dirt_l1152_115288

/-- Represents the speed of a truck on different road types -/
structure TruckSpeed where
  dirt : ℝ
  paved : ℝ

/-- Represents the travel time and distance for a truck journey -/
structure TruckJourney where
  speed : TruckSpeed
  time_dirt : ℝ
  time_paved : ℝ
  total_distance : ℝ

/-- Calculates the total distance traveled given a TruckJourney -/
def total_distance (j : TruckJourney) : ℝ :=
  j.speed.dirt * j.time_dirt + j.speed.paved * j.time_paved

/-- Theorem stating the speed of the truck on the dirt road -/
theorem truck_speed_on_dirt (j : TruckJourney) 
  (h1 : j.total_distance = 200)
  (h2 : j.time_paved = 2)
  (h3 : j.time_dirt = 3)
  (h4 : j.speed.paved = j.speed.dirt + 20) :
  j.speed.dirt = 32 := by
  sorry

#check truck_speed_on_dirt

end truck_speed_on_dirt_l1152_115288


namespace polygon_side_containment_l1152_115291

/-- A polygon is a set of points in the plane. -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A line in the plane. -/
def Line : Type := Set (ℝ × ℝ)

/-- The number of sides of a polygon. -/
def numSides (p : Polygon) : ℕ := sorry

/-- A line contains a side of a polygon. -/
def containsSide (l : Line) (p : Polygon) : Prop := sorry

/-- A line contains exactly one side of a polygon. -/
def containsExactlyOneSide (l : Line) (p : Polygon) : Prop := sorry

/-- Main theorem about 13-sided polygons and polygons with more than 13 sides. -/
theorem polygon_side_containment :
  (∀ p : Polygon, numSides p = 13 → ∃ l : Line, containsExactlyOneSide l p) ∧
  (∀ n : ℕ, n > 13 → ∃ p : Polygon, numSides p = n ∧ 
    ∀ l : Line, containsSide l p → ¬containsExactlyOneSide l p) :=
sorry

end polygon_side_containment_l1152_115291


namespace imaginary_part_of_z_l1152_115247

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by sorry

end imaginary_part_of_z_l1152_115247


namespace expression_evaluation_l1152_115269

theorem expression_evaluation :
  let a : ℤ := -2
  3 * a * (2 * a^2 - 4 * a + 3) - 2 * a^2 * (3 * a + 4) = -98 :=
by sorry

end expression_evaluation_l1152_115269


namespace mathematics_encoding_l1152_115240

def encode (c : Char) : ℕ :=
  match c with
  | 'M' => 22
  | 'A' => 32
  | 'T' => 33
  | 'E' => 11
  | 'I' => 23
  | 'K' => 13
  | _   => 0

def encodeWord (s : String) : List ℕ :=
  s.toList.map encode

theorem mathematics_encoding :
  encodeWord "MATHEMATICS" = [22, 32, 33, 11, 22, 32, 33, 23, 13, 32] :=
by sorry

end mathematics_encoding_l1152_115240


namespace petrol_price_equation_l1152_115290

/-- The original price of petrol satisfies the equation relating to a 15% price reduction and additional 7 gallons for $300 -/
theorem petrol_price_equation (P : ℝ) : P > 0 → 300 / (0.85 * P) = 300 / P + 7 := by
  sorry

end petrol_price_equation_l1152_115290


namespace range_of_k_for_special_function_l1152_115241

theorem range_of_k_for_special_function (f : ℝ → ℝ) (k a b : ℝ) :
  (∀ x, f x = Real.sqrt (x + 2) + k) →
  a < b →
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) →
  k ∈ Set.Ioo (-9/4) (-2) := by
sorry

end range_of_k_for_special_function_l1152_115241


namespace power_of_product_l1152_115204

theorem power_of_product (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end power_of_product_l1152_115204


namespace planes_perpendicular_to_line_are_parallel_l1152_115242

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  ∃ k : ℝ, p1.normal = k • p2.normal

/-- A plane is perpendicular to a line -/
def is_perpendicular_to (p : Plane3D) (l : Line3D) : Prop :=
  ∃ k : ℝ, p.normal = k • l.direction

/-- Theorem: Two planes perpendicular to the same line are parallel -/
theorem planes_perpendicular_to_line_are_parallel (p1 p2 : Plane3D) (l : Line3D)
  (h1 : is_perpendicular_to p1 l) (h2 : is_perpendicular_to p2 l) :
  are_parallel p1 p2 :=
sorry

end planes_perpendicular_to_line_are_parallel_l1152_115242


namespace winnie_lollipops_l1152_115286

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops :
  let cherry := 32
  let wintergreen := 105
  let grape := 7
  let shrimp := 198
  let friends := 12
  let total := cherry + wintergreen + grape + shrimp
  lollipops_kept total friends = 6 := by sorry

end winnie_lollipops_l1152_115286


namespace complex_calculation_l1152_115283

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = -2 + I) :
  4*a - 2*b = 16 + 6*I := by
  sorry

end complex_calculation_l1152_115283


namespace angle_abc_measure_l1152_115225

theorem angle_abc_measure (angle_cbd angle_abd angle_abc : ℝ) 
  (h1 : angle_cbd = 90)
  (h2 : angle_abd = 60)
  (h3 : angle_abc + angle_abd + angle_cbd = 190) :
  angle_abc = 40 := by
  sorry

end angle_abc_measure_l1152_115225


namespace first_three_digits_of_quotient_l1152_115208

/-- The dividend a as a real number -/
def a : ℝ := 0.1234567891011

/-- The divisor b as a real number -/
def b : ℝ := 0.51504948

/-- Theorem stating that the first three digits of a/b are 0.239 -/
theorem first_three_digits_of_quotient (ha : a > 0) (hb : b > 0) :
  0.239 * b ≤ a ∧ a < 0.24 * b :=
sorry

end first_three_digits_of_quotient_l1152_115208


namespace mike_notebooks_count_l1152_115229

theorem mike_notebooks_count :
  ∀ (total_spent blue_cost : ℕ) (red_count green_count : ℕ) (red_cost green_cost : ℕ),
    total_spent = 37 →
    red_count = 3 →
    green_count = 2 →
    red_cost = 4 →
    green_cost = 2 →
    blue_cost = 3 →
    total_spent = red_count * red_cost + green_count * green_cost + 
      ((total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost) * blue_cost →
    red_count + green_count + (total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost = 12 :=
by
  sorry

end mike_notebooks_count_l1152_115229


namespace statement_relationship_l1152_115252

theorem statement_relationship :
  (∀ x : ℝ, x^2 - 5*x < 0 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ x^2 - 5*x ≥ 0) :=
sorry

end statement_relationship_l1152_115252


namespace rectangle_diagonal_parts_l1152_115249

theorem rectangle_diagonal_parts (m n : ℕ) (hm : m = 1000) (hn : n = 1979) :
  m + n - Nat.gcd m n = 2978 := by
  sorry

end rectangle_diagonal_parts_l1152_115249


namespace committee_probability_l1152_115205

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_one_of_each : ℚ := 1705 / 1771

theorem committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_one_gender := Nat.choose boys committee_size + Nat.choose girls committee_size
  (1 : ℚ) - (all_one_gender : ℚ) / (total_committees : ℚ) = probability_at_least_one_of_each :=
sorry

end committee_probability_l1152_115205


namespace greatest_value_of_a_l1152_115299

noncomputable def f (a : ℝ) : ℝ :=
  (5 * Real.sqrt ((2 * a) ^ 2 + 1) - 4 * a ^ 2 - 2 * a) / (Real.sqrt (1 + 4 * a ^ 2) + 5)

theorem greatest_value_of_a :
  ∃ (a_max : ℝ), a_max = Real.sqrt 6 ∧
  f a_max = 1 ∧
  ∀ (a : ℝ), f a = 1 → a ≤ a_max :=
sorry

end greatest_value_of_a_l1152_115299


namespace airplane_seats_theorem_l1152_115298

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 360

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 3/10

/-- Represents the fraction of total seats in Economy -/
def economy_fraction : ℚ := 6/10

theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  (business_class_fraction * total_seats) + 
  (economy_fraction * total_seats) = total_seats := by
  sorry

end airplane_seats_theorem_l1152_115298


namespace a_66_mod_55_l1152_115260

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a_66 is congruent to 51 modulo 55 -/
theorem a_66_mod_55 : a 66 ≡ 51 [ZMOD 55] := by
  sorry

end a_66_mod_55_l1152_115260


namespace sandwich_count_l1152_115219

theorem sandwich_count (sandwich_price : ℚ) (soda_price : ℚ) (soda_count : ℕ) (total_cost : ℚ) :
  sandwich_price = 149/100 →
  soda_price = 87/100 →
  soda_count = 4 →
  total_cost = 646/100 →
  ∃ (sandwich_count : ℕ), sandwich_count = 2 ∧ 
    sandwich_count * sandwich_price + soda_count * soda_price = total_cost :=
by sorry

end sandwich_count_l1152_115219


namespace smallest_possible_a_l1152_115201

theorem smallest_possible_a (a b c : ℚ) :
  a > 0 ∧
  (∃ n : ℚ, a + b + c = n) ∧
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 2/3 = a * (x - 1/3)^2) →
  ∀ a' : ℚ, (a' > 0 ∧
    (∃ b' c' : ℚ, (∃ n : ℚ, a' + b' + c' = n) ∧
    (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 2/3 = a' * (x - 1/3)^2))) →
  a ≤ a' ∧ a = 3/8 :=
by sorry

end smallest_possible_a_l1152_115201


namespace sum_9_equals_126_l1152_115217

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + a 8 = 15 + a 5

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The theorem to be proved -/
theorem sum_9_equals_126 (seq : ArithmeticSequence) : sum_n seq 9 = 126 := by
  sorry

end sum_9_equals_126_l1152_115217


namespace f_is_increasing_on_reals_l1152_115230

-- Define the function
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem f_is_increasing_on_reals :
  (∀ x, x ∈ Set.univ → f x ∈ Set.univ) ∧
  (∀ x y, x < y → f x < f y) :=
sorry

end f_is_increasing_on_reals_l1152_115230


namespace solve_equation_solve_system_l1152_115294

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 1) / 3 - 1 = (x - 1) / 2 → x = -1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x - y = 1 ∧ 3 * x + y = 7 → x = 2 ∧ y = 1 := by sorry

end solve_equation_solve_system_l1152_115294


namespace fraction_product_simplification_l1152_115216

theorem fraction_product_simplification : 
  (2 : ℚ) / 3 * 3 / 4 * 4 / 5 * 5 / 6 = 1 / 3 := by
  sorry

end fraction_product_simplification_l1152_115216


namespace largest_prime_factor_87_l1152_115203

def numbers : List Nat := [65, 87, 143, 169, 187]

def largest_prime_factor (n : Nat) : Nat :=
  Nat.factors n |>.foldl max 0

theorem largest_prime_factor_87 :
  ∀ n ∈ numbers, n ≠ 87 → largest_prime_factor n < largest_prime_factor 87 :=
by sorry

end largest_prime_factor_87_l1152_115203


namespace equation_solutions_l1152_115285

theorem equation_solutions :
  (∃ x : ℝ, x - 4 = -5 ∧ x = -1) ∧
  (∃ x : ℝ, (1/2) * x + 2 = 6 ∧ x = 8) :=
by sorry

end equation_solutions_l1152_115285


namespace book_pages_theorem_l1152_115258

def pages_read_day1 (total : ℕ) : ℕ :=
  total / 4 + 20

def pages_left_day1 (total : ℕ) : ℕ :=
  total - pages_read_day1 total

def pages_read_day2 (total : ℕ) : ℕ :=
  (pages_left_day1 total) / 3 + 25

def pages_left_day2 (total : ℕ) : ℕ :=
  pages_left_day1 total - pages_read_day2 total

def pages_read_day3 (total : ℕ) : ℕ :=
  (pages_left_day2 total) / 2 + 30

def pages_left_day3 (total : ℕ) : ℕ :=
  pages_left_day2 total - pages_read_day3 total

theorem book_pages_theorem :
  pages_left_day3 480 = 70 := by
  sorry

end book_pages_theorem_l1152_115258


namespace complex_fraction_difference_l1152_115222

theorem complex_fraction_difference (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a - b = -1 := by sorry

end complex_fraction_difference_l1152_115222


namespace salmon_migration_l1152_115284

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) :
  male_salmon + female_salmon = 971639 := by
  sorry

end salmon_migration_l1152_115284


namespace french_only_students_l1152_115262

/-- Given a group of students with the following properties:
  * There are 28 students in total
  * Some students take French
  * 10 students take Spanish
  * 4 students take both French and Spanish
  * 13 students take neither French nor Spanish
  * Students taking both languages are not counted with those taking only French or only Spanish
This theorem proves that exactly 1 student is taking only French. -/
theorem french_only_students (total : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 28)
  (h_spanish : spanish = 10)
  (h_both : both = 4)
  (h_neither : neither = 13) :
  total - spanish - both - neither = 1 := by
sorry

end french_only_students_l1152_115262


namespace inequality_proof_l1152_115209

theorem inequality_proof (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : 0 < a₁) (h2 : 0 < a₂) (h3 : a₁ > a₂) (h4 : b₁ ≥ a₁) (h5 : b₁ * b₂ ≥ a₁ * a₂) : 
  b₁ + b₂ ≥ a₁ + a₂ := by
  sorry

end inequality_proof_l1152_115209


namespace root_in_interval_l1152_115270

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 5 = 0) → a = 3 := by
sorry

end root_in_interval_l1152_115270


namespace grain_spilled_calculation_l1152_115287

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled_calculation : original_grain - remaining_grain = 49952 := by
  sorry

end grain_spilled_calculation_l1152_115287


namespace vector_simplification_l1152_115221

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (A B C : V) : 
  (B - A) - (C - A) + (C - B) = (0 : V) := by sorry

end vector_simplification_l1152_115221


namespace legs_on_ground_for_ten_horses_l1152_115215

/-- Represents the number of legs walking on the ground given the conditions of the problem --/
def legs_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 10 horses, there are 50 legs walking on the ground --/
theorem legs_on_ground_for_ten_horses :
  legs_on_ground 10 = 50 := by
  sorry

end legs_on_ground_for_ten_horses_l1152_115215


namespace james_matches_count_l1152_115213

/-- The number of boxes in a dozen -/
def boxesPerDozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def dozensOfBoxes : ℕ := 5

/-- The number of matches in each box -/
def matchesPerBox : ℕ := 20

/-- Theorem: Given the conditions, James has 1200 matches -/
theorem james_matches_count :
  dozensOfBoxes * boxesPerDozen * matchesPerBox = 1200 := by
  sorry

end james_matches_count_l1152_115213


namespace library_fiction_percentage_l1152_115266

theorem library_fiction_percentage 
  (total_volumes : ℕ) 
  (fiction_percentage : ℚ)
  (transfer_fraction : ℚ)
  (fiction_transfer_fraction : ℚ)
  (h1 : total_volumes = 18360)
  (h2 : fiction_percentage = 30 / 100)
  (h3 : transfer_fraction = 1 / 3)
  (h4 : fiction_transfer_fraction = 1 / 5) :
  let original_fiction := (fiction_percentage * total_volumes : ℚ)
  let transferred_volumes := (transfer_fraction * total_volumes : ℚ)
  let transferred_fiction := (fiction_transfer_fraction * transferred_volumes : ℚ)
  let remaining_fiction := original_fiction - transferred_fiction
  let remaining_volumes := total_volumes - transferred_volumes
  (remaining_fiction / remaining_volumes) * 100 = 35 := by
sorry


end library_fiction_percentage_l1152_115266


namespace valid_sequences_count_l1152_115246

/-- The number of colors available at each station -/
def num_colors : ℕ := 4

/-- The number of stations (including start and end) -/
def num_stations : ℕ := 4

/-- A function that calculates the number of valid color sequences -/
def count_valid_sequences : ℕ :=
  num_colors * (num_colors - 1)^(num_stations - 1)

/-- Theorem stating that the number of valid color sequences is 108 -/
theorem valid_sequences_count :
  count_valid_sequences = 108 := by sorry

end valid_sequences_count_l1152_115246


namespace quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l1152_115276

-- Problem 1
theorem quadratic_equation_1 (x : ℝ) : 
  (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) → x^2 - 2*x - 2 = 0 := by sorry

-- Problem 2
theorem quadratic_equation_2 (x : ℝ) :
  (x = -4 ∨ x = 1) → (x + 4)^2 = 5*(x + 4) := by sorry

-- Problem 3
theorem quadratic_equation_3 (x : ℝ) :
  (x = (-3 + 2*Real.sqrt 6) / 3 ∨ x = (-3 - 2*Real.sqrt 6) / 3) → 3*x^2 + 6*x - 5 = 0 := by sorry

-- Problem 4
theorem quadratic_equation_4 (x : ℝ) :
  (x = (-1 + Real.sqrt 5) / 4 ∨ x = (-1 - Real.sqrt 5) / 4) → 4*x^2 + 2*x = 1 := by sorry

end quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l1152_115276


namespace fourth_root_is_four_l1152_115223

/-- The polynomial with coefficients c and d -/
def polynomial (c d x : ℝ) : ℝ :=
  c * x^4 + (c + 3*d) * x^3 + (d - 4*c) * x^2 + (10 - c) * x + (5 - 2*d)

/-- The theorem stating that if -1, 2, and -3 are roots of the polynomial,
    then 4 is the fourth root -/
theorem fourth_root_is_four (c d : ℝ) :
  polynomial c d (-1) = 0 →
  polynomial c d 2 = 0 →
  polynomial c d (-3) = 0 →
  ∃ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ -3 ∧ polynomial c d x = 0 ∧ x = 4 := by
  sorry

end fourth_root_is_four_l1152_115223


namespace formula_satisfies_table_l1152_115243

def table : List (ℕ × ℕ) := [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]

theorem formula_satisfies_table : ∀ (pair : ℕ × ℕ), pair ∈ table → (pair.2 : ℚ) = (pair.1 : ℚ) ^ 3 := by
  sorry

end formula_satisfies_table_l1152_115243


namespace equation_solution_l1152_115282

theorem equation_solution : 
  Real.sqrt (1 + Real.sqrt (2 + Real.sqrt 49)) = (1 + Real.sqrt 49) ^ (1/3) := by
  sorry

end equation_solution_l1152_115282


namespace wendy_recycling_points_l1152_115207

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points : 
  (total_bags - unrecycled_bags) * points_per_bag = 45 := by
  sorry

end wendy_recycling_points_l1152_115207


namespace father_age_equals_sum_of_brothers_ages_l1152_115200

/-- Represents the current ages of the family members -/
structure FamilyAges where
  ivan : ℕ
  vincent : ℕ
  jakub : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.vincent = 11 ∧
  ages.jakub = 9 ∧
  ages.ivan = 5 * (ages.jakub / 3) ∧
  ages.father = 3 * ages.ivan

/-- The theorem to be proved -/
theorem father_age_equals_sum_of_brothers_ages (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ∃ (n : ℕ), n = 5 ∧ 
  ages.father + n = ages.ivan + ages.vincent + ages.jakub + 3 * n :=
sorry

end father_age_equals_sum_of_brothers_ages_l1152_115200


namespace brass_weight_l1152_115224

theorem brass_weight (copper_ratio : ℚ) (zinc_ratio : ℚ) (zinc_weight : ℚ) : 
  copper_ratio = 3 → 
  zinc_ratio = 7 → 
  zinc_weight = 70 → 
  (copper_ratio + zinc_ratio) * (zinc_weight / zinc_ratio) = 100 :=
by sorry

end brass_weight_l1152_115224


namespace games_sale_value_l1152_115264

def initial_cost : ℝ := 200
def value_multiplier : ℝ := 3
def sold_percentage : ℝ := 0.4

theorem games_sale_value :
  let new_value := initial_cost * value_multiplier
  let sold_value := new_value * sold_percentage
  sold_value = 240 := by
  sorry

end games_sale_value_l1152_115264


namespace triangle_DEF_angle_F_l1152_115295

theorem triangle_DEF_angle_F (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  2 * Real.sin D + 3 * Real.cos E = 3 →
  3 * Real.sin E + 5 * Real.cos D = 4 →
  Real.sin F = 1/4 := by
sorry

end triangle_DEF_angle_F_l1152_115295


namespace product_equals_eight_l1152_115245

theorem product_equals_eight :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end product_equals_eight_l1152_115245


namespace jims_remaining_distance_l1152_115289

theorem jims_remaining_distance 
  (total_distance : ℕ) 
  (driven_distance : ℕ) 
  (b_to_c : ℕ) 
  (c_to_d : ℕ) 
  (d_to_e : ℕ) 
  (h1 : total_distance = 2500) 
  (h2 : driven_distance = 642) 
  (h3 : b_to_c = 400) 
  (h4 : c_to_d = 550) 
  (h5 : d_to_e = 200) : 
  total_distance - driven_distance = b_to_c + c_to_d + d_to_e :=
by sorry

end jims_remaining_distance_l1152_115289


namespace inequality_preservation_l1152_115228

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end inequality_preservation_l1152_115228


namespace endpoint_coordinate_sum_l1152_115232

/-- Given a line segment with one endpoint (5, -2) and midpoint (3, 4),
    the sum of coordinates of the other endpoint is 11. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 + x) / 2 = 3 → 
    (-2 + y) / 2 = 4 → 
    x + y = 11 := by
  sorry

end endpoint_coordinate_sum_l1152_115232


namespace cars_meeting_time_l1152_115236

theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) : 
  (highway_length / (speed1 + speed2)) = 3 := by
sorry

end cars_meeting_time_l1152_115236


namespace alex_corn_purchase_l1152_115214

/-- The price of corn per pound -/
def corn_price : ℝ := 1.20

/-- The price of beans per pound -/
def bean_price : ℝ := 0.60

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 27.00

/-- The amount of corn bought in pounds -/
def corn_amount : ℝ := 15.0

theorem alex_corn_purchase :
  ∃ (bean_amount : ℝ),
    corn_amount + bean_amount = total_pounds ∧
    corn_price * corn_amount + bean_price * bean_amount = total_cost :=
by
  sorry

end alex_corn_purchase_l1152_115214


namespace rachel_brownies_l1152_115271

/-- Rachel's brownie problem -/
theorem rachel_brownies (total : ℕ) (brought_to_school : ℕ) (left_at_home : ℕ) : 
  total = 40 → brought_to_school = 16 → left_at_home = total - brought_to_school →
  left_at_home = 24 := by
  sorry

end rachel_brownies_l1152_115271


namespace find_taco_order_l1152_115234

/-- Represents the number of tacos and enchiladas in an order -/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order in dollars -/
def cost (order : Order) (taco_price enchilada_price : ℝ) : ℝ :=
  taco_price * order.tacos + enchilada_price * order.enchiladas

theorem find_taco_order : ∃ (my_order : Order) (enchilada_price : ℝ),
  my_order.enchiladas = 3 ∧
  cost my_order 0.9 enchilada_price = 7.8 ∧
  cost (Order.mk 3 5) 0.9 enchilada_price = 12.7 ∧
  my_order.tacos = 2 := by
  sorry

end find_taco_order_l1152_115234


namespace ellipse_dist_to_directrix_l1152_115275

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The distance from a point to the left focus of an ellipse -/
def distToLeftFocus (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- The distance from a point to the right directrix of an ellipse -/
def distToRightDirectrix (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- Theorem: For the given ellipse, if a point on the ellipse is at distance 8 from the left focus,
    then its distance to the right directrix is 5/2 -/
theorem ellipse_dist_to_directrix (E : Ellipse) (P : PointOnEllipse E) :
  E.a = 5 ∧ E.b = 3 ∧ distToLeftFocus E P = 8 → distToRightDirectrix E P = 5/2 := by
  sorry

end ellipse_dist_to_directrix_l1152_115275


namespace min_races_for_top_five_l1152_115253

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 4 horses. -/
structure Race :=
  (participants : Finset Horse)
  (hLimited : participants.card ≤ 4)

/-- Represents the outcome of a series of races. -/
structure RaceOutcome :=
  (races : List Race)
  (topFive : Finset Horse)
  (hTopFiveSize : topFive.card = 5)

/-- The main theorem stating the minimum number of races required. -/
theorem min_races_for_top_five (horses : Finset Horse) 
  (hSize : horses.card = 30) :
  ∃ (outcome : RaceOutcome), 
    outcome.topFive ⊆ horses ∧ 
    outcome.races.length = 8 ∧ 
    (∀ (alt_outcome : RaceOutcome), 
      alt_outcome.topFive ⊆ horses → 
      alt_outcome.races.length ≥ 8) :=
sorry

end min_races_for_top_five_l1152_115253


namespace min_sum_of_reciprocal_line_l1152_115231

theorem min_sum_of_reciprocal_line (a b : ℝ) : 
  a > 0 → b > 0 → (1 : ℝ) / a + (1 : ℝ) / b = 1 → (a + b) ≥ 4 := by
  sorry

end min_sum_of_reciprocal_line_l1152_115231


namespace tetrahedron_vertices_prove_tetrahedron_vertices_l1152_115233

/-- A tetrahedron is a three-dimensional polyhedron with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_vertices

/-- Prove that a tetrahedron has 4 vertices. -/
theorem prove_tetrahedron_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end tetrahedron_vertices_prove_tetrahedron_vertices_l1152_115233


namespace max_intersections_convex_polygons_l1152_115218

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  isConvex : Bool

/-- Represents the state of two polygons after rotation -/
structure RotatedPolygons (Q1 Q2 : ConvexPolygon) where
  canIntersect : Bool

/-- Calculates the maximum number of intersections between two rotated polygons -/
def maxIntersections (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2) : ℕ :=
  if state.canIntersect then Q1.sides * Q2.sides else 0

theorem max_intersections_convex_polygons :
  ∀ (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2),
    Q1.sides = 5 →
    Q2.sides = 7 →
    Q1.isConvex = true →
    Q2.isConvex = true →
    state.canIntersect = true →
    maxIntersections Q1 Q2 state = 35 := by
  sorry

end max_intersections_convex_polygons_l1152_115218


namespace smallest_number_with_remainder_two_l1152_115273

theorem smallest_number_with_remainder_two : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (∀ m : ℕ, m > 1 → (m % 3 = 2) → (m % 4 = 2) → (m % 5 = 2) → m ≥ n) ∧
  (n = 62) := by
  sorry

end smallest_number_with_remainder_two_l1152_115273


namespace evenly_spaced_poles_l1152_115251

/-- Given five evenly spaced poles along a straight road, 
    if the distance between the second and fifth poles is 90 feet, 
    then the distance between the first and fifth poles is 120 feet. -/
theorem evenly_spaced_poles (n : ℕ) (d : ℝ) (h1 : n = 5) (h2 : d = 90) :
  let pole_distance (i j : ℕ) := d * (j - i) / 3
  pole_distance 1 5 = 120 := by
  sorry

end evenly_spaced_poles_l1152_115251


namespace zero_points_sum_gt_one_l1152_115238

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : Real.log x₁ + 1 / (2 * x₁) = m) 
  (h₃ : Real.log x₂ + 1 / (2 * x₂) = m) : 
  x₁ + x₂ > 1 := by
  sorry

end zero_points_sum_gt_one_l1152_115238


namespace average_temperature_l1152_115210

def temperatures : List ℚ := [73, 76, 75, 78, 74]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 75.2 := by
  sorry

end average_temperature_l1152_115210


namespace dans_balloons_l1152_115206

theorem dans_balloons (fred_balloons sam_balloons total_balloons : ℕ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : total_balloons = 72) :
  total_balloons - (fred_balloons + sam_balloons) = 16 := by
  sorry

end dans_balloons_l1152_115206


namespace solve_cubic_equation_l1152_115267

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 3)^3 = (1/27)⁻¹ ∧ y = 6 :=
by
  sorry

end solve_cubic_equation_l1152_115267


namespace sum_of_smallest_multiples_l1152_115292

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := sorry

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := sorry

/-- c is a two-digit number -/
axiom c_two_digit : 10 ≤ c ∧ c ≤ 99

/-- d is a three-digit number -/
axiom d_three_digit : 100 ≤ d ∧ d ≤ 999

/-- c is a multiple of 5 -/
axiom c_multiple_of_5 : ∃ k : ℕ, c = 5 * k

/-- d is a multiple of 7 -/
axiom d_multiple_of_7 : ∃ k : ℕ, d = 7 * k

/-- c is the smallest two-digit multiple of 5 -/
axiom c_smallest : ∀ x : ℕ, (10 ≤ x ∧ x ≤ 99 ∧ ∃ k : ℕ, x = 5 * k) → c ≤ x

/-- d is the smallest three-digit multiple of 7 -/
axiom d_smallest : ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ ∃ k : ℕ, x = 7 * k) → d ≤ x

theorem sum_of_smallest_multiples : c + d = 115 := by sorry

end sum_of_smallest_multiples_l1152_115292


namespace sweet_cookies_eaten_indeterminate_l1152_115272

def initial_salty_cookies : ℕ := 26
def initial_sweet_cookies : ℕ := 17
def salty_cookies_eaten : ℕ := 9
def salty_cookies_left : ℕ := 17

theorem sweet_cookies_eaten_indeterminate :
  ∀ (sweet_cookies_eaten : ℕ),
    sweet_cookies_eaten ≤ initial_sweet_cookies →
    salty_cookies_left = initial_salty_cookies - salty_cookies_eaten →
    ∃ (sweet_cookies_eaten' : ℕ),
      sweet_cookies_eaten' ≠ sweet_cookies_eaten ∧
      sweet_cookies_eaten' ≤ initial_sweet_cookies :=
by sorry

end sweet_cookies_eaten_indeterminate_l1152_115272


namespace equation_transformation_l1152_115235

theorem equation_transformation (m : ℝ) : 2 * m - 1 = 3 → 2 * m = 3 + 1 := by
  sorry

end equation_transformation_l1152_115235


namespace trivia_team_distribution_l1152_115244

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) 
  (h1 : total = 58) 
  (h2 : not_picked = 10) 
  (h3 : groups = 8) :
  (total - not_picked) / groups = 6 := by
  sorry

end trivia_team_distribution_l1152_115244


namespace hyperbola_eccentricity_l1152_115256

/-- Given a hyperbola C₁ and a parabola C₂ in the Cartesian coordinate system (xOy):
    C₁: x²/a² - y²/b² = 1 (a > 0, b > 0)
    C₂: x² = 2py (p > 0)
    
    The asymptotes of C₁ intersect with C₂ at points O, A, B.
    The orthocenter of triangle OAB is the focus of C₂.

    This theorem states that the eccentricity of C₁ is 3/2. -/
theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) : 
  let C₁ := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let C₂ := fun (x y : ℝ) => x^2 = 2*p*y
  let asymptotes := fun (x y : ℝ) => y = (b/a)*x ∨ y = -(b/a)*x
  let O := (0, 0)
  let A := (2*p*b/a, 2*p*b^2/a^2)
  let B := (-2*p*b/a, 2*p*b^2/a^2)
  let focus := (0, p/2)
  let orthocenter := focus
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  (∀ x y, asymptotes x y → C₂ x y → (x = 0 ∨ x = 2*p*b/a ∨ x = -2*p*b/a)) →
  (orthocenter = focus) →
  eccentricity = 3/2 := by
sorry

end hyperbola_eccentricity_l1152_115256


namespace icosahedral_die_expected_digits_l1152_115211

/-- The expected number of digits when rolling a fair icosahedral die -/
def expected_digits : ℝ := 1.55

/-- The number of faces on an icosahedral die -/
def num_faces : ℕ := 20

/-- The number of one-digit faces on the die -/
def one_digit_faces : ℕ := 9

/-- The number of two-digit faces on the die -/
def two_digit_faces : ℕ := 11

theorem icosahedral_die_expected_digits :
  expected_digits = (one_digit_faces : ℝ) / num_faces + 2 * (two_digit_faces : ℝ) / num_faces :=
sorry

end icosahedral_die_expected_digits_l1152_115211


namespace dryer_price_difference_dryer_costs_less_l1152_115265

/-- Given a washing machine price of $100 and a dryer with an unknown price,
    if there's a 10% discount on the total and the final price is $153,
    then the dryer costs $30 less than the washing machine. -/
theorem dryer_price_difference (dryer_price : ℝ) : 
  (100 + dryer_price) * 0.9 = 153 → dryer_price = 70 :=
by
  sorry

/-- The difference in price between the washing machine and the dryer -/
def price_difference : ℝ := 100 - 70

theorem dryer_costs_less : price_difference = 30 :=
by
  sorry

end dryer_price_difference_dryer_costs_less_l1152_115265


namespace inequality_proof_l1152_115202

theorem inequality_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (1.7 : ℝ)^(0.3 : ℝ) > (0.9 : ℝ)^(3.1 : ℝ) := by
  sorry

end inequality_proof_l1152_115202


namespace min_value_expression_l1152_115278

theorem min_value_expression (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_upper_bound : a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3)
  (h_sum1 : a + b + c + d = 6)
  (h_sum2 : e + f = 2) :
  (Real.sqrt (a^2 + 4) + Real.sqrt (b^2 + e^2) + Real.sqrt (c^2 + f^2) + Real.sqrt (d^2 + 4))^2 ≥ 72 :=
by sorry

end min_value_expression_l1152_115278


namespace simplify_expression_l1152_115237

theorem simplify_expression (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (2 - m) / (2 + m) := by
  sorry

end simplify_expression_l1152_115237


namespace sixth_power_sum_l1152_115250

theorem sixth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 := by
  sorry

end sixth_power_sum_l1152_115250


namespace greatest_three_digit_multiple_of_17_l1152_115280

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l1152_115280


namespace circles_tangent_m_values_l1152_115254

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 8*x + 8*y + m = 0

-- Define tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  (∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' m → (x' = x ∧ y' = y))

-- Theorem statement
theorem circles_tangent_m_values :
  ∀ m : ℝ, are_tangent m ↔ (m = -4 ∨ m = 16) :=
sorry

end circles_tangent_m_values_l1152_115254


namespace intersection_A_complement_B_l1152_115296

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {2, 3, 4}

-- Define set B
def B : Finset Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_A_complement_B_l1152_115296


namespace equation_solution_l1152_115268

theorem equation_solution : 
  ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ (x = 2) :=
by
  sorry

end equation_solution_l1152_115268


namespace platform_length_calculation_l1152_115248

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 14 seconds, prove that the length of the platform
    is approximately 535.77 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 14) :
  ∃ platform_length : ℝ, abs (platform_length - 535.77) < 0.01 :=
by
  sorry


end platform_length_calculation_l1152_115248


namespace alcohol_concentration_in_mixture_l1152_115297

structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

def totalAlcohol (vessels : List Vessel) : ℝ :=
  vessels.foldl (fun acc v => acc + v.capacity * v.alcoholConcentration) 0

def largeContainerCapacity : ℝ := 25

theorem alcohol_concentration_in_mixture 
  (vessels : List Vessel)
  (h1 : vessels = [
    ⟨2, 0.3⟩, 
    ⟨6, 0.4⟩, 
    ⟨4, 0.25⟩, 
    ⟨3, 0.35⟩, 
    ⟨5, 0.2⟩
  ]) :
  (totalAlcohol vessels) / largeContainerCapacity = 0.242 := by
  sorry

end alcohol_concentration_in_mixture_l1152_115297


namespace opposite_reciprocal_sum_l1152_115277

theorem opposite_reciprocal_sum (a b c d : ℝ) (m : ℕ) : 
  b ≠ 0 →
  a = -b →
  c * d = 1 →
  m < 2 →
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -2 ∨ 
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -1 :=
by sorry

end opposite_reciprocal_sum_l1152_115277


namespace water_fraction_after_replacements_l1152_115281

-- Define the radiator capacity
def radiator_capacity : ℚ := 20

-- Define the volume replaced each time
def replacement_volume : ℚ := 5

-- Define the number of replacements
def num_replacements : ℕ := 5

-- Define the fraction of liquid remaining after each replacement
def remaining_fraction : ℚ := (radiator_capacity - replacement_volume) / radiator_capacity

-- Statement of the problem
theorem water_fraction_after_replacements :
  (remaining_fraction ^ num_replacements : ℚ) = 243 / 1024 := by
  sorry

end water_fraction_after_replacements_l1152_115281
