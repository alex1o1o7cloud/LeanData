import Mathlib

namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2635_263534

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (3 * x^2 - 2 * x) / ((x - 4) * (x - 2)^2) = 
  10 / (x - 4) + (-7) / (x - 2) + (-4) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2635_263534


namespace NUMINAMATH_CALUDE_hash_five_two_l2635_263510

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

-- Theorem statement
theorem hash_five_two : hash 5 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_two_l2635_263510


namespace NUMINAMATH_CALUDE_jan_ian_distance_difference_l2635_263502

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  t : ℝ  -- Ian's driving time
  s : ℝ  -- Ian's driving speed
  ian_distance : ℝ := t * s
  han_distance : ℝ := (t + 2) * (s + 10)
  jan_distance : ℝ := (t + 3) * (s + 15)

/-- The theorem stating the difference between Jan's and Ian's distances -/
theorem jan_ian_distance_difference (scenario : DrivingScenario) 
  (h : scenario.han_distance = scenario.ian_distance + 100) : 
  scenario.jan_distance - scenario.ian_distance = 165 := by
  sorry

#check jan_ian_distance_difference

end NUMINAMATH_CALUDE_jan_ian_distance_difference_l2635_263502


namespace NUMINAMATH_CALUDE_special_polynomial_value_l2635_263522

theorem special_polynomial_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5*x^6 + x^2 = 8436*x - 338 := by
sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l2635_263522


namespace NUMINAMATH_CALUDE_skittles_distribution_l2635_263507

theorem skittles_distribution (initial_skittles : ℕ) (additional_skittles : ℕ) (num_people : ℕ) :
  initial_skittles = 14 →
  additional_skittles = 22 →
  num_people = 7 →
  (initial_skittles + additional_skittles) / num_people = 5 :=
by sorry

end NUMINAMATH_CALUDE_skittles_distribution_l2635_263507


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2635_263501

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2635_263501


namespace NUMINAMATH_CALUDE_prove_z_value_l2635_263556

theorem prove_z_value (z : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 2.650793650793651) → 
  z = 1.00 := by
sorry

end NUMINAMATH_CALUDE_prove_z_value_l2635_263556


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_sqrt_two_over_two_l2635_263575

theorem cosine_sine_sum_equals_sqrt_two_over_two : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_sqrt_two_over_two_l2635_263575


namespace NUMINAMATH_CALUDE_equation_solution_l2635_263578

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x - 2) = 2 * x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 0 ∧ x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2635_263578


namespace NUMINAMATH_CALUDE_girls_percentage_less_than_boys_l2635_263536

theorem girls_percentage_less_than_boys :
  ∀ (girls boys : ℝ),
  boys = girls * 1.25 →
  (girls - boys) / boys * 100 = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_less_than_boys_l2635_263536


namespace NUMINAMATH_CALUDE_vector_linear_combination_l2635_263547

/-- Given vectors a, b, and c in R², prove that c is a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, -2)) : 
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l2635_263547


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2635_263500

def isGeometricSequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem geometric_sequence_product (a b : ℝ) :
  isGeometricSequence 2 a b 16 → a * b = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2635_263500


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l2635_263521

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l2635_263521


namespace NUMINAMATH_CALUDE_tuesday_kids_count_l2635_263541

def monday_kids : ℕ := 12
def total_kids : ℕ := 19

theorem tuesday_kids_count : total_kids - monday_kids = 7 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_count_l2635_263541


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_20_l2635_263516

/-- The sum of an arithmetic sequence modulo 20 -/
theorem arithmetic_sequence_sum_mod_20 : ∃ m : ℕ, 
  let a := 2  -- first term
  let l := 102  -- last term
  let d := 5  -- common difference
  let n := (l - a) / d + 1  -- number of terms
  let S := n * (a + l) / 2  -- sum of arithmetic sequence
  S % 20 = m ∧ m < 20 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_20_l2635_263516


namespace NUMINAMATH_CALUDE_tournament_games_l2635_263599

theorem tournament_games (x : ℕ) : 
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * x = x ∧ 
  (2 / 3 : ℚ) * (x + 10) + (1 / 3 : ℚ) * (x + 10) = x + 10 ∧
  (2 / 3 : ℚ) * (x + 10) = (3 / 4 : ℚ) * x + 5 ∧
  (1 / 3 : ℚ) * (x + 10) = (1 / 4 : ℚ) * x + 5 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_tournament_games_l2635_263599


namespace NUMINAMATH_CALUDE_fence_building_time_l2635_263562

/-- The time in minutes to build one fence -/
def time_per_fence : ℕ := 30

/-- The number of fences initially built -/
def initial_fences : ℕ := 10

/-- The total number of fences after additional work -/
def total_fences : ℕ := 26

/-- The additional work time in hours -/
def additional_work_time : ℕ := 8

/-- Theorem stating that the time per fence is 30 minutes -/
theorem fence_building_time :
  time_per_fence = 30 ∧
  initial_fences = 10 ∧
  total_fences = 26 ∧
  additional_work_time = 8 ∧
  (total_fences - initial_fences) * time_per_fence = additional_work_time * 60 :=
by sorry

end NUMINAMATH_CALUDE_fence_building_time_l2635_263562


namespace NUMINAMATH_CALUDE_complex_sum_equals_minus_ten_i_l2635_263540

theorem complex_sum_equals_minus_ten_i :
  (5 - 5 * Complex.I) + (-2 - Complex.I) - (3 + 4 * Complex.I) = -10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_minus_ten_i_l2635_263540


namespace NUMINAMATH_CALUDE_not_square_for_prime_l2635_263542

theorem not_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) : ¬∃ (a : ℤ), (7 * p + 3^p - 4 : ℤ) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_for_prime_l2635_263542


namespace NUMINAMATH_CALUDE_smallest_number_l2635_263558

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The binary representation of 1111 --/
def binary_1111 : List Nat := [1, 1, 1, 1]

/-- The base-6 representation of 210 --/
def base6_210 : List Nat := [2, 1, 0]

/-- The base-4 representation of 1000 --/
def base4_1000 : List Nat := [1, 0, 0, 0]

/-- The octal representation of 101 --/
def octal_101 : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_1111 2 < to_decimal base6_210 6 ∧
  to_decimal binary_1111 2 < to_decimal base4_1000 4 ∧
  to_decimal binary_1111 2 < to_decimal octal_101 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2635_263558


namespace NUMINAMATH_CALUDE_range_of_a_l2635_263523

theorem range_of_a (a x : ℝ) : 
  (∀ x, (x^2 - 7*x + 10 ≤ 0 → a < x ∧ x < a + 1) ∧ 
        (a < x ∧ x < a + 1 → ¬(∀ y, a < y ∧ y < a + 1 → y^2 - 7*y + 10 ≤ 0))) →
  2 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2635_263523


namespace NUMINAMATH_CALUDE_sarahs_trip_length_l2635_263509

theorem sarahs_trip_length :
  ∀ (x : ℝ),
  (x / 4 : ℝ) + 15 + (x / 3 : ℝ) = x →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_sarahs_trip_length_l2635_263509


namespace NUMINAMATH_CALUDE_intersection_point_in_zero_one_l2635_263596

-- Define the function f(x) = x^3 - (1/2)^x
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^x

-- State the theorem
theorem intersection_point_in_zero_one :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_in_zero_one_l2635_263596


namespace NUMINAMATH_CALUDE_f_increasing_l2635_263560

-- Define the function f(x) = x^3 + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_increasing : ∀ (a b : ℝ), a < b → f a < f b := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l2635_263560


namespace NUMINAMATH_CALUDE_nicky_running_time_l2635_263574

/-- Proves that Nicky runs for 60 seconds before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 1500)
  (h2 : head_start = 25)
  (h3 : cristina_speed = 6)
  (h4 : nicky_speed = 3.5) :
  ∃ (t : ℝ), t = 60 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end NUMINAMATH_CALUDE_nicky_running_time_l2635_263574


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l2635_263515

theorem fraction_product_equals_one : 
  (2 * 7) / (14 * 6) * (6 * 14) / (2 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l2635_263515


namespace NUMINAMATH_CALUDE_mika_stickers_decoration_l2635_263584

/-- The number of stickers Mika used to decorate the greeting card -/
def stickers_used_for_decoration (initial : ℕ) (bought : ℕ) (received : ℕ) (given_away : ℕ) (left : ℕ) : ℕ :=
  initial + bought + received - given_away - left

/-- Theorem stating that Mika used 58 stickers to decorate the greeting card -/
theorem mika_stickers_decoration :
  stickers_used_for_decoration 20 26 20 6 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_decoration_l2635_263584


namespace NUMINAMATH_CALUDE_mens_average_weight_l2635_263518

theorem mens_average_weight (num_men : ℕ) (num_women : ℕ) (avg_women : ℝ) (avg_total : ℝ) :
  num_men = 8 →
  num_women = 6 →
  avg_women = 120 →
  avg_total = 160 →
  let total_people := num_men + num_women
  let avg_men := (avg_total * total_people - avg_women * num_women) / num_men
  avg_men = 190 := by
sorry

end NUMINAMATH_CALUDE_mens_average_weight_l2635_263518


namespace NUMINAMATH_CALUDE_total_frogs_is_18_l2635_263550

/-- The number of frogs inside the pond -/
def frogs_inside : ℕ := 12

/-- The number of frogs outside the pond -/
def frogs_outside : ℕ := 6

/-- The total number of frogs -/
def total_frogs : ℕ := frogs_inside + frogs_outside

/-- Theorem stating that the total number of frogs is 18 -/
theorem total_frogs_is_18 : total_frogs = 18 := by sorry

end NUMINAMATH_CALUDE_total_frogs_is_18_l2635_263550


namespace NUMINAMATH_CALUDE_right_side_difference_l2635_263581

/-- A triangle with specific side lengths -/
structure Triangle where
  left : ℝ
  right : ℝ
  base : ℝ

/-- The properties of our specific triangle -/
def special_triangle (t : Triangle) : Prop :=
  t.left = 12 ∧ 
  t.base = 24 ∧ 
  t.left + t.right + t.base = 50 ∧
  t.right > t.left

theorem right_side_difference (t : Triangle) (h : special_triangle t) : 
  t.right - t.left = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_side_difference_l2635_263581


namespace NUMINAMATH_CALUDE_harry_joe_fish_ratio_l2635_263531

/-- Proves that Harry has 4 times as many fish as Joe given the conditions -/
theorem harry_joe_fish_ratio :
  ∀ (harry joe sam : ℕ),
  joe = 8 * sam →
  sam = 7 →
  harry = 224 →
  harry = 4 * joe :=
by
  sorry

end NUMINAMATH_CALUDE_harry_joe_fish_ratio_l2635_263531


namespace NUMINAMATH_CALUDE_simplify_trig_expression_1_simplify_trig_expression_2_l2635_263549

-- Part 1
theorem simplify_trig_expression_1 :
  (Real.sin (35 * π / 180))^2 - (1/2) = -Real.cos (10 * π / 180) * Real.cos (80 * π / 180) := by
  sorry

-- Part 2
theorem simplify_trig_expression_2 (α : ℝ) :
  (1 / Real.tan (α/2) - Real.tan (α/2)) * ((1 - Real.cos (2*α)) / Real.sin (2*α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_1_simplify_trig_expression_2_l2635_263549


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l2635_263538

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a binary number to its decimal equivalent -/
def binaryToDecimal (bin : BinaryNumber) : ℕ :=
  bin.enum.foldl (fun acc (i, digit) =>
    acc + match digit with
      | BinaryDigit.zero => 0
      | BinaryDigit.one => 2^i
  ) 0

/-- The binary representation of 1101 -/
def bin1101 : BinaryNumber :=
  [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.one]

theorem binary_1101_equals_13 :
  binaryToDecimal bin1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l2635_263538


namespace NUMINAMATH_CALUDE_loris_books_needed_l2635_263535

/-- The number of books needed by Loris to match Lamont's count -/
def books_needed (darryl_books lamont_books loris_books total_books : ℕ) : ℕ :=
  lamont_books - loris_books

/-- Theorem stating the number of books Loris needs -/
theorem loris_books_needed 
  (darryl_books : ℕ) 
  (lamont_books : ℕ) 
  (loris_books : ℕ) 
  (total_books : ℕ) 
  (h1 : darryl_books = 20) 
  (h2 : lamont_books = 2 * darryl_books) 
  (h3 : total_books = loris_books + darryl_books + lamont_books) 
  (h4 : total_books = 97) : 
  books_needed darryl_books lamont_books loris_books total_books = 3 := by
  sorry

#eval books_needed 20 40 37 97

end NUMINAMATH_CALUDE_loris_books_needed_l2635_263535


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_six_is_solution_six_is_smallest_l2635_263568

theorem smallest_integer_fraction (x : ℤ) : x > 5 ∧ (x^2 - 4*x + 13) % (x - 5) = 0 → x ≥ 6 := by
  sorry

theorem six_is_solution : (6^2 - 4*6 + 13) % (6 - 5) = 0 := by
  sorry

theorem six_is_smallest : ∀ (y : ℤ), y > 5 ∧ y < 6 → (y^2 - 4*y + 13) % (y - 5) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_six_is_solution_six_is_smallest_l2635_263568


namespace NUMINAMATH_CALUDE_neutron_electron_difference_l2635_263528

/-- Represents an atomic element -/
structure Element where
  protonNumber : ℕ
  massNumber : ℕ

/-- Calculates the number of neutrons in an element -/
def neutronCount (e : Element) : ℕ :=
  e.massNumber - e.protonNumber

/-- The number of electrons in a neutral atom is equal to the proton number -/
def electronCount (e : Element) : ℕ :=
  e.protonNumber

/-- Theorem: For an element with proton number 118 and mass number 293,
    the difference between the number of neutrons and electrons is 57 -/
theorem neutron_electron_difference (e : Element) 
    (h1 : e.protonNumber = 118) (h2 : e.massNumber = 293) : 
    neutronCount e - electronCount e = 57 := by
  sorry

end NUMINAMATH_CALUDE_neutron_electron_difference_l2635_263528


namespace NUMINAMATH_CALUDE_floor_abs_sum_l2635_263517

theorem floor_abs_sum : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l2635_263517


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l2635_263527

/-- Represents the typing speeds of Tim and Tom -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ

/-- The total pages typed by Tim and Tom in one hour -/
def totalPages (speed : TypingSpeed) : ℝ := speed.tim + speed.tom

/-- The total pages typed when Tom increases his speed by 25% -/
def increasedTotalPages (speed : TypingSpeed) : ℝ := speed.tim + 1.25 * speed.tom

theorem typing_speed_ratio 
  (speed : TypingSpeed) 
  (h1 : totalPages speed = 12)
  (h2 : increasedTotalPages speed = 14) :
  speed.tom / speed.tim = 2 := by
  sorry

#check typing_speed_ratio

end NUMINAMATH_CALUDE_typing_speed_ratio_l2635_263527


namespace NUMINAMATH_CALUDE_alpha_values_l2635_263597

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) :
  ∃ (x y : ℝ), α = Complex.mk x y ∧ 
    ((x = (1 + 8*Real.sqrt 2/9)/2 ∨ x = (1 - 8*Real.sqrt 2/9)/2) ∧
     y^2 = 9 - ((x + 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_values_l2635_263597


namespace NUMINAMATH_CALUDE_supplements_calculation_l2635_263576

/-- The number of boxes of supplements delivered by Mr. Anderson -/
def boxes_of_supplements : ℕ := 760 - 472

/-- The total number of boxes of medicine delivered -/
def total_boxes : ℕ := 760

/-- The number of boxes of vitamins delivered -/
def vitamin_boxes : ℕ := 472

theorem supplements_calculation :
  boxes_of_supplements = 288 :=
by sorry

end NUMINAMATH_CALUDE_supplements_calculation_l2635_263576


namespace NUMINAMATH_CALUDE_two_point_distribution_properties_l2635_263567

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  X : ℝ → ℝ
  prob_zero : ℝ
  prob_one : ℝ
  sum_to_one : prob_zero + prob_one = 1
  only_two_points : ∀ x, X x ≠ 0 → X x = 1

/-- Expected value of a two-point distribution -/
def expected_value (dist : TwoPointDistribution) : ℝ :=
  0 * dist.prob_zero + 1 * dist.prob_one

/-- Variance of a two-point distribution -/
def variance (dist : TwoPointDistribution) : ℝ :=
  dist.prob_zero * (0 - expected_value dist)^2 + 
  dist.prob_one * (1 - expected_value dist)^2

/-- Theorem: Expected value and variance for a specific two-point distribution -/
theorem two_point_distribution_properties (dist : TwoPointDistribution)
  (h : dist.prob_zero = 1/4) :
  expected_value dist = 3/4 ∧ variance dist = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_properties_l2635_263567


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2635_263582

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) : 
  r = 6 → sector_angle = 300 → 
  2 * π * r * (360 - sector_angle) / 360 = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2635_263582


namespace NUMINAMATH_CALUDE_min_value_theorem_l2635_263585

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2635_263585


namespace NUMINAMATH_CALUDE_square_room_tiles_l2635_263555

theorem square_room_tiles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive size
  (2 * n = 62) →  -- Total tiles on both diagonals
  n * n = 961  -- Total tiles in the room
  := by sorry

end NUMINAMATH_CALUDE_square_room_tiles_l2635_263555


namespace NUMINAMATH_CALUDE_turnover_equation_l2635_263571

-- Define the monthly average growth rate
variable (x : ℝ)

-- Define the initial turnover in January (in units of 10,000 yuan)
def initial_turnover : ℝ := 200

-- Define the total turnover in the first quarter (in units of 10,000 yuan)
def total_turnover : ℝ := 1000

-- Theorem statement
theorem turnover_equation :
  initial_turnover + initial_turnover * (1 + x) + initial_turnover * (1 + x)^2 = total_turnover := by
  sorry

end NUMINAMATH_CALUDE_turnover_equation_l2635_263571


namespace NUMINAMATH_CALUDE_square_perimeter_count_l2635_263530

/-- The number of students on each side of the square arrangement -/
def side_length : ℕ := 10

/-- The number of students on the perimeter of a square arrangement -/
def perimeter_count (n : ℕ) : ℕ := 4 * n - 4

theorem square_perimeter_count :
  perimeter_count side_length = 36 := by
  sorry


end NUMINAMATH_CALUDE_square_perimeter_count_l2635_263530


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l2635_263514

theorem graduating_class_boys_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 345 →
  difference = 69 →
  total = boys + (boys + difference) →
  boys = 138 := by
sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l2635_263514


namespace NUMINAMATH_CALUDE_function_multiple_preimages_l2635_263544

theorem function_multiple_preimages :
  ∃ (f : ℝ → ℝ) (y : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = y ∧ f x₂ = y := by
  sorry

end NUMINAMATH_CALUDE_function_multiple_preimages_l2635_263544


namespace NUMINAMATH_CALUDE_marble_remainder_l2635_263532

theorem marble_remainder (r p : ℕ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_marble_remainder_l2635_263532


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2635_263583

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value (a k c : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (a + k) * x + c
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (m = (-a^2 - 2*a*k - k^2 + 4*a*c) / (4*a)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2635_263583


namespace NUMINAMATH_CALUDE_parabola_intersection_kite_coefficient_sum_l2635_263580

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coefficientSum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_intersection_kite_coefficient_sum :
  ∀ k : Kite,
    k.p1 = Parabola.mk k.p1.a (-3) →
    k.p2 = Parabola.mk (-k.p2.a) 5 →
    kiteArea k = 15 →
    ∃ ε > 0, |coefficientSum k - 2.3| < ε :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_kite_coefficient_sum_l2635_263580


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l2635_263569

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l2635_263569


namespace NUMINAMATH_CALUDE_first_digit_change_largest_l2635_263503

def original_number : ℚ := 0.12345678

def change_digit (n : ℚ) (position : ℕ) : ℚ :=
  n + (9 - (n * 10^position % 10)) / 10^position

theorem first_digit_change_largest :
  ∀ position : ℕ, position > 0 → 
    change_digit original_number 1 ≥ change_digit original_number position :=
by sorry

end NUMINAMATH_CALUDE_first_digit_change_largest_l2635_263503


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l2635_263564

theorem unique_modular_equivalence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14525 [ZMOD 16] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l2635_263564


namespace NUMINAMATH_CALUDE_complement_of_M_l2635_263598

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set M
def M : Set ℝ := {x | Real.log (1 - x) > 0}

-- State the theorem
theorem complement_of_M : 
  (Mᶜ : Set ℝ) = {x | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2635_263598


namespace NUMINAMATH_CALUDE_right_triangle_legs_l2635_263570

theorem right_triangle_legs (area : ℝ) (hypotenuse : ℝ) (a b : ℝ) :
  area = 504 →
  hypotenuse = 65 →
  a * b = 2 * area →
  a^2 + b^2 = hypotenuse^2 →
  ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l2635_263570


namespace NUMINAMATH_CALUDE_cost_of_300_candies_l2635_263588

/-- The cost of a single candy in cents -/
def candy_cost : ℕ := 5

/-- The number of candies -/
def num_candies : ℕ := 300

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 300 candies is 15 dollars -/
theorem cost_of_300_candies :
  (num_candies * candy_cost) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_300_candies_l2635_263588


namespace NUMINAMATH_CALUDE_ap_num_terms_l2635_263508

/-- Represents an arithmetic progression with an even number of terms. -/
structure ArithmeticProgression where
  n : ℕ                   -- Number of terms
  a : ℚ                   -- First term
  d : ℚ                   -- Common difference
  n_even : Even n         -- n is even
  last_minus_first : a + (n - 1) * d - a = 16  -- Last term exceeds first by 16

/-- The sum of odd-numbered terms in the arithmetic progression. -/
def sum_odd_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + (ap.n - 2) * ap.d)

/-- The sum of even-numbered terms in the arithmetic progression. -/
def sum_even_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + 2 * ap.d + (ap.n - 2) * ap.d)

/-- Theorem stating the conditions and conclusion about the number of terms. -/
theorem ap_num_terms (ap : ArithmeticProgression) 
  (h_odd : sum_odd_terms ap = 81)
  (h_even : sum_even_terms ap = 75) : 
  ap.n = 8 := by sorry

end NUMINAMATH_CALUDE_ap_num_terms_l2635_263508


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2635_263524

theorem inequality_solution_set (x : ℝ) :
  (8 * x^3 - 6 * x^2 + 5 * x - 1 < 4) ↔ (x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2635_263524


namespace NUMINAMATH_CALUDE_solve_for_q_l2635_263579

theorem solve_for_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p*q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2635_263579


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l2635_263539

theorem solve_equation_for_x (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 80) : 
  x = 26 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l2635_263539


namespace NUMINAMATH_CALUDE_function_is_zero_l2635_263551

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that a function satisfies the given conditions. -/
def SatisfiesConditions (f : NatFunction) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℕ, x > y → f (x^2 - y^2) = f x * f y

/-- Theorem stating that any function satisfying the conditions must be identically zero. -/
theorem function_is_zero (f : NatFunction) (h : SatisfiesConditions f) : 
  ∀ x : ℕ, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_function_is_zero_l2635_263551


namespace NUMINAMATH_CALUDE_parentheses_value_l2635_263548

theorem parentheses_value (x : ℤ) (h : x - (-2) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_value_l2635_263548


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l2635_263594

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The function is monotonically increasing on ℝ -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The theorem statement -/
theorem min_value_cubic_function (a b c d : ℝ) :
  a > 0 →
  a < (2/3) * b →
  monotonically_increasing (cubic_function a b c d) →
  (c / (2 * b - 3 * a)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l2635_263594


namespace NUMINAMATH_CALUDE_complex_simplification_l2635_263587

theorem complex_simplification :
  (4 - 3 * Complex.I) * 2 - (6 - 3 * Complex.I) = 2 - 3 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2635_263587


namespace NUMINAMATH_CALUDE_rahul_to_deepak_age_ratio_l2635_263529

def rahul_age_after_6_years : ℕ := 26
def deepak_current_age : ℕ := 15
def years_to_add : ℕ := 6

theorem rahul_to_deepak_age_ratio :
  (rahul_age_after_6_years - years_to_add) / deepak_current_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_to_deepak_age_ratio_l2635_263529


namespace NUMINAMATH_CALUDE_imaginary_unit_sixth_power_l2635_263573

theorem imaginary_unit_sixth_power (i : ℂ) (hi : i * i = -1) : i^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sixth_power_l2635_263573


namespace NUMINAMATH_CALUDE_equation_solution_l2635_263533

theorem equation_solution :
  ∃ x : ℝ, -2 * (x - 1) = 4 ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2635_263533


namespace NUMINAMATH_CALUDE_total_money_l2635_263543

/-- The total amount of money A, B, and C have together is 500, given the specified conditions. -/
theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 330 → c = 30 → a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2635_263543


namespace NUMINAMATH_CALUDE_ambika_candles_l2635_263566

theorem ambika_candles (ambika : ℕ) (aniyah : ℕ) : 
  aniyah = 6 * ambika →
  (ambika + aniyah) / 2 = 14 →
  ambika = 4 := by
sorry

end NUMINAMATH_CALUDE_ambika_candles_l2635_263566


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_l2635_263593

def probability_celtics_win : ℚ := 3/4

def probability_lakers_win : ℚ := 1 - probability_celtics_win

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem lakers_win_in_seven (probability_celtics_win : ℚ) 
  (h1 : probability_celtics_win = 3/4) 
  (h2 : games_to_win = 4) 
  (h3 : total_games = 7) : 
  ℚ :=
by
  sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_l2635_263593


namespace NUMINAMATH_CALUDE_rosa_pages_called_l2635_263553

/-- The number of pages Rosa called last week -/
def last_week_pages : ℝ := 10.2

/-- The total number of pages Rosa called -/
def total_pages : ℝ := 18.8

/-- The number of pages Rosa called this week -/
def this_week_pages : ℝ := total_pages - last_week_pages

theorem rosa_pages_called : this_week_pages = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_rosa_pages_called_l2635_263553


namespace NUMINAMATH_CALUDE_total_pencils_count_l2635_263592

/-- The number of colors in the rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of pencils in a color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The number of Emily's friends who bought a color box -/
def emilys_friends : ℕ := 7

/-- The total number of pencils Emily and her friends have -/
def total_pencils : ℕ := pencils_per_box + emilys_friends * pencils_per_box

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l2635_263592


namespace NUMINAMATH_CALUDE_train_length_calculation_l2635_263520

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with speeds of 45 km/hr and 36 km/hr respectively, if the faster train passes the slower train
    in 36 seconds, then the length of each train is 45 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) 
  (h1 : faster_speed = 45) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : 
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let distance_covered := relative_speed * passing_time
  let train_length := distance_covered / 2
  train_length = 45 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2635_263520


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l2635_263525

theorem nested_expression_evaluation : (5*(5*(5*(5+1)+1)+1)+1) = 781 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l2635_263525


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2635_263511

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2635_263511


namespace NUMINAMATH_CALUDE_skittles_left_l2635_263586

def initial_skittles : ℕ := 250
def reduction_percentage : ℚ := 175 / 1000

theorem skittles_left :
  ⌊(initial_skittles : ℚ) - (initial_skittles : ℚ) * reduction_percentage⌋ = 206 :=
by
  sorry

end NUMINAMATH_CALUDE_skittles_left_l2635_263586


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l2635_263589

/-- The area of a square containing four circles of radius 7 inches, 
    arranged so that two circles fit into the width and height of the square. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l2635_263589


namespace NUMINAMATH_CALUDE_equality_condition_l2635_263506

theorem equality_condition (x y z : ℝ) : 
  x + y * z = (x + y) * (x + z) ↔ x + y + z = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2635_263506


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2635_263552

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = -b) : 
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2635_263552


namespace NUMINAMATH_CALUDE_odd_function_property_l2635_263554

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_f_1 : f 1 = 1/2)
  (h_f_shift : ∀ x, f (x + 2) = f x + f 2) :
  f 5 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2635_263554


namespace NUMINAMATH_CALUDE_birthday_on_sunday_l2635_263565

/-- Represents days of the week -/
inductive Day : Type
  | sunday : Day
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day

/-- Returns the day before the given day -/
def dayBefore (d : Day) : Day :=
  match d with
  | Day.sunday => Day.saturday
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday

/-- Returns the day that is n days after the given day -/
def daysAfter (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => daysAfter (match d with
    | Day.sunday => Day.monday
    | Day.monday => Day.tuesday
    | Day.tuesday => Day.wednesday
    | Day.wednesday => Day.thursday
    | Day.thursday => Day.friday
    | Day.friday => Day.saturday
    | Day.saturday => Day.sunday) m

theorem birthday_on_sunday (today : Day) (birthday : Day) : 
  today = Day.thursday → 
  daysAfter (dayBefore birthday) 2 = dayBefore (dayBefore (dayBefore today)) → 
  birthday = Day.sunday := by
  sorry

end NUMINAMATH_CALUDE_birthday_on_sunday_l2635_263565


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2635_263519

theorem triangle_perimeter (a b x : ℝ) : 
  a = 1 → b = 2 → x^2 - 3*x + 2 = 0 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) →
  a + b + x = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2635_263519


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2635_263559

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -3 < x → x < 3 → x ≠ -1 → x ≠ 1 → x ≠ 0 →
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x + 1) - x + 1) = -1 / x) ∧
  (x = -2 → -1 / x = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2635_263559


namespace NUMINAMATH_CALUDE_repair_cost_is_correct_l2635_263526

/-- Calculates the total cost of car repair given the following conditions:
  * Two mechanics work on the car
  * First mechanic: $60/hour, 8 hours/day, 14 days
  * Second mechanic: $75/hour, 6 hours/day, 10 days
  * 15% discount on first mechanic's labor cost
  * 10% discount on second mechanic's labor cost
  * Parts cost: $3,200
  * 7% sales tax on final bill after discounts
-/
def totalRepairCost (
  mechanic1_rate : ℝ)
  (mechanic1_hours : ℝ)
  (mechanic1_days : ℝ)
  (mechanic2_rate : ℝ)
  (mechanic2_hours : ℝ)
  (mechanic2_days : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (parts_cost : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let mechanic1_cost := mechanic1_rate * mechanic1_hours * mechanic1_days
  let mechanic2_cost := mechanic2_rate * mechanic2_hours * mechanic2_days
  let discounted_mechanic1_cost := mechanic1_cost * (1 - discount1)
  let discounted_mechanic2_cost := mechanic2_cost * (1 - discount2)
  let total_before_tax := discounted_mechanic1_cost + discounted_mechanic2_cost + parts_cost
  total_before_tax * (1 + sales_tax_rate)

/-- Theorem stating that the total repair cost is $13,869.34 given the specific conditions -/
theorem repair_cost_is_correct :
  totalRepairCost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_correct_l2635_263526


namespace NUMINAMATH_CALUDE_friends_contribution_proof_l2635_263545

def check_amount : ℝ := 200
def tip_percentage : ℝ := 0.20
def marks_contribution : ℝ := 30

theorem friends_contribution_proof :
  ∃ (friend_contribution : ℝ),
    tip_percentage * check_amount = friend_contribution + marks_contribution ∧
    friend_contribution = 10 := by
  sorry

end NUMINAMATH_CALUDE_friends_contribution_proof_l2635_263545


namespace NUMINAMATH_CALUDE_product_inspection_probability_l2635_263590

theorem product_inspection_probability : 
  let p_good_as_defective : ℝ := 0.02
  let p_defective_as_good : ℝ := 0.01
  let num_good : ℕ := 3
  let num_defective : ℕ := 1
  let p_correct_good : ℝ := 1 - p_good_as_defective
  let p_correct_defective : ℝ := 1 - p_defective_as_good
  (p_correct_good ^ num_good) * (p_correct_defective ^ num_defective) = 0.932 :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_probability_l2635_263590


namespace NUMINAMATH_CALUDE_smallest_lattice_triangle_area_is_half_l2635_263557

/-- A lattice triangle is a triangle on a square grid where all vertices are grid points. -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The area of a grid square is 1 square unit. -/
def grid_square_area : ℝ := 1

/-- The area of a lattice triangle -/
def lattice_triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- The smallest possible area of a lattice triangle -/
def smallest_lattice_triangle_area : ℝ := sorry

/-- Theorem: The area of the smallest lattice triangle is 1/2 square unit -/
theorem smallest_lattice_triangle_area_is_half :
  smallest_lattice_triangle_area = 1/2 := by sorry

end NUMINAMATH_CALUDE_smallest_lattice_triangle_area_is_half_l2635_263557


namespace NUMINAMATH_CALUDE_odd_number_divisibility_l2635_263591

theorem odd_number_divisibility (a : ℕ) (h_odd : Odd a) :
  (∀ m : ℕ, ∃ (k : ℕ → ℕ), Function.Injective k ∧ ∀ i : ℕ, (a^(k i) - 1) % (2^m) = 0) ∧
  (∃ (S : Finset ℕ), ∀ m : ℕ, (a^m - 1) % (2^m) = 0 → m ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_divisibility_l2635_263591


namespace NUMINAMATH_CALUDE_trains_crossing_time_l2635_263513

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 400)
  (h3 : speed1 = 36 * 1000 / 3600)
  (h4 : speed2 = 18 * 1000 / 3600)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0) :
  (length1 + length2) / (speed1 + speed2) = 46.67 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l2635_263513


namespace NUMINAMATH_CALUDE_boys_on_playground_l2635_263512

/-- The number of girls on the playground -/
def num_girls : ℕ := 28

/-- The difference between the number of boys and girls -/
def difference : ℕ := 7

/-- The number of boys on the playground -/
def num_boys : ℕ := num_girls + difference

theorem boys_on_playground : num_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_playground_l2635_263512


namespace NUMINAMATH_CALUDE_fruit_basket_composition_l2635_263546

/-- Represents a fruit basket -/
structure FruitBasket where
  apples : ℕ
  pears : ℕ
  others : ℕ

/-- The total number of fruits in the basket -/
def FruitBasket.total (b : FruitBasket) : ℕ := b.apples + b.pears + b.others

/-- Predicate to check if any 3 fruits contain an apple -/
def hasAppleIn3 (b : FruitBasket) : Prop :=
  b.pears + b.others ≤ 2

/-- Predicate to check if any 4 fruits contain a pear -/
def hasPearIn4 (b : FruitBasket) : Prop :=
  b.apples + b.others ≤ 3

/-- The main theorem -/
theorem fruit_basket_composition (b : FruitBasket) :
  b.total ≥ 5 →
  hasAppleIn3 b →
  hasPearIn4 b →
  b.apples = 3 ∧ b.pears = 2 ∧ b.others = 0 :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_composition_l2635_263546


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2635_263577

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = (-1/5 : ℂ) - (3/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2635_263577


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l2635_263563

/-- Given a function f(x) = kx³ - 3(k+1)x² - k² + 1 where k > 0,
    if the decreasing interval of f(x) is (0, 4), then k = 1. -/
theorem function_decreasing_interval (k : ℝ) (h₁ : k > 0) :
  let f : ℝ → ℝ := λ x => k * x^3 - 3 * (k + 1) * x^2 - k^2 + 1
  (∀ x ∈ Set.Ioo 0 4, ∀ y ∈ Set.Ioo 0 4, x < y → f x > f y) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l2635_263563


namespace NUMINAMATH_CALUDE_profit_percentage_l2635_263561

theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : 
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2635_263561


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l2635_263572

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

theorem vectors_perpendicular : 
  let c := (a.1 - b.1, a.2 - b.2)
  (a.1 * c.1 + a.2 * c.2 = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l2635_263572


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l2635_263537

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 given boat and stream speeds -/
theorem rowing_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 60) 
  (h2 : stream_speed = 20) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check rowing_time_ratio

end NUMINAMATH_CALUDE_rowing_time_ratio_l2635_263537


namespace NUMINAMATH_CALUDE_train_length_l2635_263595

/-- The length of a train given crossing times -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 180)
  (h3 : platform_length = 600)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  (platform_time * platform_length) / (platform_time - tree_time) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2635_263595


namespace NUMINAMATH_CALUDE_two_times_zero_times_one_plus_one_l2635_263504

theorem two_times_zero_times_one_plus_one : 2 * 0 * 1 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_times_zero_times_one_plus_one_l2635_263504


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l2635_263505

def prime_factors : List (Nat × Nat) := [(2, 12), (3, 16), (7, 18), (11, 7)]

def count_square_factors (p : Nat) (e : Nat) : Nat :=
  (e / 2) + 1

theorem count_perfect_square_factors :
  (prime_factors.map (fun (p, e) => count_square_factors p e)).prod = 2520 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l2635_263505
