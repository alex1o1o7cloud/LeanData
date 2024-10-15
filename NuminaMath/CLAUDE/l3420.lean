import Mathlib

namespace NUMINAMATH_CALUDE_division_of_fractions_l3420_342071

theorem division_of_fractions : (3/8) / (5/12) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3420_342071


namespace NUMINAMATH_CALUDE_circular_permutation_divisibility_l3420_342080

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def circular_permutation (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, k < 5 ∧ m = (n * 10^k) % 100000 + n / (100000 / 10^k)}

theorem circular_permutation_divisibility (n : ℕ) (h1 : is_five_digit n) (h2 : n % 41 = 0) :
  ∀ m ∈ circular_permutation n, m % 41 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circular_permutation_divisibility_l3420_342080


namespace NUMINAMATH_CALUDE_college_student_count_l3420_342009

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, 
    the total number of students is 494 -/
theorem college_student_count (c : College) 
    (h1 : c.boys * 5 = c.girls * 8) 
    (h2 : c.girls = 190) : 
  c.total = 494 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l3420_342009


namespace NUMINAMATH_CALUDE_parabola_point_relation_l3420_342011

theorem parabola_point_relation (a y₁ y₂ y₃ : ℝ) :
  a < -1 →
  y₁ = (a - 1)^2 →
  y₂ = a^2 →
  y₃ = (a + 1)^2 →
  y₁ > y₂ ∧ y₂ > y₃ :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l3420_342011


namespace NUMINAMATH_CALUDE_coffee_doughnut_problem_l3420_342095

theorem coffee_doughnut_problem :
  ∀ (c d : ℕ),
    c + d = 7 →
    (90 * c + 60 * d) % 100 = 0 →
    c = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_doughnut_problem_l3420_342095


namespace NUMINAMATH_CALUDE_initial_distance_between_cars_l3420_342042

/-- Proves that the initial distance between two cars is 16 miles given their speeds and overtaking time -/
theorem initial_distance_between_cars
  (speed_A : ℝ)
  (speed_B : ℝ)
  (overtake_time : ℝ)
  (ahead_distance : ℝ)
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : overtake_time = 3)
  (h4 : ahead_distance = 8) :
  speed_A * overtake_time - speed_B * overtake_time - ahead_distance = 16 :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_between_cars_l3420_342042


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l3420_342076

theorem x_plus_y_equals_negative_one 
  (x y : ℝ) 
  (h1 : x + |x| + y = 5) 
  (h2 : x + |y| - y = 6) : 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l3420_342076


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l3420_342036

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (x' y' : ℝ), x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l3420_342036


namespace NUMINAMATH_CALUDE_second_group_factories_l3420_342010

theorem second_group_factories (total : ℕ) (first_group : ℕ) (unchecked : ℕ) :
  total = 169 →
  first_group = 69 →
  unchecked = 48 →
  total - (first_group + unchecked) = 52 := by
  sorry

end NUMINAMATH_CALUDE_second_group_factories_l3420_342010


namespace NUMINAMATH_CALUDE_max_amount_received_back_l3420_342031

/-- Represents the denominations of chips --/
inductive ChipDenomination
  | twoHundred
  | fiveHundred

/-- Represents the number of chips lost for each denomination --/
structure ChipsLost where
  twoHundred : ℕ
  fiveHundred : ℕ

def totalChipsBought : ℕ := 50000

def chipValue (d : ChipDenomination) : ℕ :=
  match d with
  | ChipDenomination.twoHundred => 200
  | ChipDenomination.fiveHundred => 500

def totalChipsLost (c : ChipsLost) : ℕ := c.twoHundred + c.fiveHundred

def validChipsLost (c : ChipsLost) : Prop :=
  totalChipsLost c = 25 ∧
  (c.twoHundred = c.fiveHundred + 5 ∨ c.twoHundred + 5 = c.fiveHundred)

def valueLost (c : ChipsLost) : ℕ :=
  c.twoHundred * chipValue ChipDenomination.twoHundred +
  c.fiveHundred * chipValue ChipDenomination.fiveHundred

def amountReceivedBack (c : ChipsLost) : ℕ := totalChipsBought - valueLost c

theorem max_amount_received_back :
  ∃ (c : ChipsLost), validChipsLost c ∧
    (∀ (c' : ChipsLost), validChipsLost c' → amountReceivedBack c ≥ amountReceivedBack c') ∧
    amountReceivedBack c = 42000 :=
  sorry

end NUMINAMATH_CALUDE_max_amount_received_back_l3420_342031


namespace NUMINAMATH_CALUDE_min_value_trig_function_l3420_342002

theorem min_value_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * (Real.cos x)^2 - Real.sin (2 * x)
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = 1 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l3420_342002


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l3420_342020

-- Define the points A, B, C, and X
variable (A B C X : ℝ × ℝ)

-- Define the lengths of the sides
def AB : ℝ := 80
def AC : ℝ := 36
def BC : ℝ := 72

-- Define the angle bisector property
def is_angle_bisector (A B C X : ℝ × ℝ) : Prop :=
  (A.1 - X.1) * (C.2 - X.2) = (C.1 - X.1) * (A.2 - X.2) ∧
  (B.1 - X.1) * (C.2 - X.2) = (C.1 - X.1) * (B.2 - X.2)

-- State the theorem
theorem angle_bisector_theorem (h : is_angle_bisector A B C X) :
  dist A X = 80 / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l3420_342020


namespace NUMINAMATH_CALUDE_joyce_apples_l3420_342067

theorem joyce_apples (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 75 → given = 52 → remaining = initial - given → remaining = 23 := by
sorry

end NUMINAMATH_CALUDE_joyce_apples_l3420_342067


namespace NUMINAMATH_CALUDE_max_min_difference_c_l3420_342051

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15) :
  (3 : ℝ) - (-7/3) = 16/3 := by sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l3420_342051


namespace NUMINAMATH_CALUDE_rectangle_area_l3420_342003

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 250 → l * w = 2500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3420_342003


namespace NUMINAMATH_CALUDE_total_growing_space_l3420_342050

/-- Represents a garden bed with length and width dimensions -/
structure GardenBed where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℕ := bed.length * bed.width

/-- Calculates the total area of multiple identical garden beds -/
def totalArea (bed : GardenBed) (count : ℕ) : ℕ := area bed * count

/-- The set of garden beds Amy is building -/
def amysGardenBeds : List (GardenBed × ℕ) := [
  (⟨5, 4⟩, 3),
  (⟨6, 3⟩, 4),
  (⟨7, 5⟩, 2),
  (⟨8, 4⟩, 1)
]

/-- Theorem stating that the total growing space is 234 sq ft -/
theorem total_growing_space :
  (amysGardenBeds.map (fun (bed, count) => totalArea bed count)).sum = 234 := by
  sorry

end NUMINAMATH_CALUDE_total_growing_space_l3420_342050


namespace NUMINAMATH_CALUDE_second_person_share_l3420_342055

/-- Represents the share of money for each person -/
structure Shares :=
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ)

/-- Theorem: Given a sum of money distributed among four people in the proportion 6:3:5:4,
    where the third person gets 1000 more than the fourth, the second person's share is 3000 -/
theorem second_person_share
  (shares : Shares)
  (h1 : shares.a = 6 * shares.d)
  (h2 : shares.b = 3 * shares.d)
  (h3 : shares.c = 5 * shares.d)
  (h4 : shares.c = shares.d + 1000) :
  shares.b = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_second_person_share_l3420_342055


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3420_342016

theorem smallest_common_multiple_of_6_and_15 : 
  ∃ b : ℕ+, (∀ m : ℕ+, (6 ∣ m) ∧ (15 ∣ m) → b ≤ m) ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l3420_342016


namespace NUMINAMATH_CALUDE_boyd_boys_percentage_l3420_342033

/-- Represents the number of friends on a social media platform -/
structure SocialMediaFriends where
  boys : ℕ
  girls : ℕ

/-- Represents a person's friends on different social media platforms -/
structure Person where
  facebook : SocialMediaFriends
  instagram : SocialMediaFriends

def Julian : Person :=
  { facebook := { boys := 48, girls := 32 },
    instagram := { boys := 45, girls := 105 } }

def Boyd : Person :=
  { facebook := { boys := 1, girls := 64 },
    instagram := { boys := 135, girls := 0 } }

def total_friends (p : Person) : ℕ :=
  p.facebook.boys + p.facebook.girls + p.instagram.boys + p.instagram.girls

def boys_percentage (p : Person) : ℚ :=
  (p.facebook.boys + p.instagram.boys : ℚ) / total_friends p

theorem boyd_boys_percentage :
  boys_percentage Boyd = 68 / 100 :=
sorry

end NUMINAMATH_CALUDE_boyd_boys_percentage_l3420_342033


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3420_342053

theorem inequality_equivalence (x : ℝ) : 
  (x - 2) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3420_342053


namespace NUMINAMATH_CALUDE_median_less_than_half_sum_of_sides_l3420_342069

/-- Given a triangle ABC with sides a, b, and c, and median CM₃ to side c,
    prove that CM₃ < (a + b) / 2 -/
theorem median_less_than_half_sum_of_sides 
  {a b c : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : c < a + b) :
  let CM₃ := Real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  CM₃ < (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_median_less_than_half_sum_of_sides_l3420_342069


namespace NUMINAMATH_CALUDE_property_transaction_outcome_l3420_342014

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

theorem property_transaction_outcome :
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 862.50 := by sorry

end NUMINAMATH_CALUDE_property_transaction_outcome_l3420_342014


namespace NUMINAMATH_CALUDE_sine_equation_equality_l3420_342021

theorem sine_equation_equality (α β γ τ : ℝ) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0) 
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) : 
  α = γ ∨ α = τ := by
  sorry

end NUMINAMATH_CALUDE_sine_equation_equality_l3420_342021


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3420_342086

theorem cubic_polynomial_coefficient (a b c d : ℝ) : 
  let g := fun x => a * x^3 + b * x^2 + c * x + d
  (g (-2) = 0) → (g 0 = 0) → (g 2 = 0) → (g 1 = 3) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3420_342086


namespace NUMINAMATH_CALUDE_coaching_fee_calculation_l3420_342017

/-- Calculates the number of days from January 1 to a given date in a non-leap year -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  match month with
  | 1 => day
  | 2 => 31 + day
  | 3 => 59 + day
  | 4 => 90 + day
  | 5 => 120 + day
  | 6 => 151 + day
  | 7 => 181 + day
  | 8 => 212 + day
  | 9 => 243 + day
  | 10 => 273 + day
  | 11 => 304 + day
  | 12 => 334 + day
  | _ => 0

/-- Daily coaching charge in dollars -/
def dailyCharge : Nat := 39

/-- Calculates the total coaching fee -/
def totalCoachingFee (startMonth : Nat) (startDay : Nat) (endMonth : Nat) (endDay : Nat) : Nat :=
  let totalDays := daysFromNewYear endMonth endDay - daysFromNewYear startMonth startDay + 1
  totalDays * dailyCharge

theorem coaching_fee_calculation :
  totalCoachingFee 1 1 11 3 = 11934 := by
  sorry

#eval totalCoachingFee 1 1 11 3

end NUMINAMATH_CALUDE_coaching_fee_calculation_l3420_342017


namespace NUMINAMATH_CALUDE_houses_built_during_boom_l3420_342019

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558

theorem houses_built_during_boom : 
  current_houses - original_houses = 97741 := by sorry

end NUMINAMATH_CALUDE_houses_built_during_boom_l3420_342019


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_power_minus_75_l3420_342061

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_power_minus_75 :
  sum_of_digits (10^50 - 75) = 439 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_power_minus_75_l3420_342061


namespace NUMINAMATH_CALUDE_larger_number_problem_l3420_342097

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 35 → max x y = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3420_342097


namespace NUMINAMATH_CALUDE_total_apples_l3420_342035

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples : pinky_apples + danny_apples = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l3420_342035


namespace NUMINAMATH_CALUDE_walking_speed_problem_l3420_342078

/-- Proves that given the conditions of the walking problem, Deepak's speed is 4.5 km/hr -/
theorem walking_speed_problem (track_circumference : ℝ) (wife_speed : ℝ) (meeting_time : ℝ) :
  track_circumference = 528 →
  wife_speed = 3.75 →
  meeting_time = 3.84 →
  ∃ (deepak_speed : ℝ),
    deepak_speed = 4.5 ∧
    (wife_speed * 1000 / 60) * meeting_time + deepak_speed * 1000 / 60 * meeting_time = track_circumference :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3420_342078


namespace NUMINAMATH_CALUDE_hot_chocolate_max_servings_l3420_342001

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ
  servings : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings possible given a recipe and available ingredients -/
def max_servings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem hot_chocolate_max_servings :
  let recipe : Recipe := {
    chocolate := 3,
    sugar := 1/3,
    water := 3/2,
    milk := 5,
    servings := 6
  }
  let ingredients : Ingredients := {
    chocolate := 8,
    sugar := 3,
    milk := 12
  }
  max_servings recipe ingredients = 16 := by sorry

end NUMINAMATH_CALUDE_hot_chocolate_max_servings_l3420_342001


namespace NUMINAMATH_CALUDE_base5_324_equals_binary_1011001_l3420_342045

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to binary --/
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

/-- Theorem: The base-5 number 324₍₅₎ is equal to the binary number 1011001₍₂₎ --/
theorem base5_324_equals_binary_1011001 :
  decimalToBinary (base5ToDecimal [4, 2, 3]) = [1, 0, 1, 1, 0, 0, 1] := by
  sorry


end NUMINAMATH_CALUDE_base5_324_equals_binary_1011001_l3420_342045


namespace NUMINAMATH_CALUDE_marked_hexagon_properties_l3420_342089

/-- A regular hexagon with diagonals marked -/
structure MarkedHexagon where
  /-- The area of the hexagon in square centimeters -/
  area : ℝ
  /-- The hexagon is regular -/
  regular : Bool
  /-- All diagonals are marked -/
  diagonals_marked : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : MarkedHexagon) : ℕ := sorry

/-- The area of the smaller hexagon formed by quadrilateral parts -/
def smaller_hexagon_area (h : MarkedHexagon) : ℝ := sorry

/-- Theorem about the properties of a marked regular hexagon -/
theorem marked_hexagon_properties (h : MarkedHexagon) 
  (h_area : h.area = 144)
  (h_regular : h.regular = true)
  (h_marked : h.diagonals_marked = true) :
  num_parts h = 24 ∧ smaller_hexagon_area h = 48 := by sorry

end NUMINAMATH_CALUDE_marked_hexagon_properties_l3420_342089


namespace NUMINAMATH_CALUDE_place_two_before_eq_l3420_342046

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_lt_10 : hundreds < 10
  t_lt_10 : tens < 10
  u_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Represents the operation of placing 2 before a three-digit number -/
def place_two_before (n : ThreeDigitNumber) : ℕ :=
  2000 + 100 * n.hundreds + 10 * n.tens + n.units

/-- Theorem stating that placing 2 before a three-digit number results in 2000 + 100h + 10t + u -/
theorem place_two_before_eq (n : ThreeDigitNumber) :
  place_two_before n = 2000 + 100 * n.hundreds + 10 * n.tens + n.units := by
  sorry

end NUMINAMATH_CALUDE_place_two_before_eq_l3420_342046


namespace NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3420_342083

/-- Given a natural number, return the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2007 and whose sum of digits equals 2007 -/
theorem exists_multiple_with_digit_sum :
  ∃ n : ℕ, (∃ k : ℕ, n = k * 2007) ∧ sumOfDigits n = 2007 := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3420_342083


namespace NUMINAMATH_CALUDE_circular_course_circumference_l3420_342075

/-- The circumference of a circular course where two people walking at different speeds meet after a certain time. -/
theorem circular_course_circumference
  (speed_a speed_b : ℝ)
  (meeting_time : ℝ)
  (h1 : speed_a = 4)
  (h2 : speed_b = 5)
  (h3 : meeting_time = 115)
  (h4 : speed_b > speed_a) :
  (speed_b - speed_a) * meeting_time = 115 :=
by sorry

end NUMINAMATH_CALUDE_circular_course_circumference_l3420_342075


namespace NUMINAMATH_CALUDE_parallel_tangent_length_l3420_342077

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  base : ℝ
  height : ℝ
  inscribed_circle : Circle

/-- Represents a tangent line to the inscribed circle, parallel to the base -/
structure ParallelTangent where
  triangle : IsoscelesTriangleWithInscribedCircle
  length : ℝ

/-- The theorem statement -/
theorem parallel_tangent_length 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (tangent : ParallelTangent) 
  (h1 : triangle.base = 12)
  (h2 : triangle.height = 8)
  (h3 : tangent.triangle = triangle) : 
  tangent.length = 3 := by sorry

end NUMINAMATH_CALUDE_parallel_tangent_length_l3420_342077


namespace NUMINAMATH_CALUDE_area_equality_iff_rectangle_l3420_342062

/-- A quadrilateral with sides a, b, c, d and area A -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  A : ℝ

/-- Definition of a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ (w h : ℝ), q.a = w ∧ q.b = h ∧ q.c = w ∧ q.d = h ∧ q.A = w * h

/-- Theorem: Area equality holds iff the quadrilateral is a rectangle -/
theorem area_equality_iff_rectangle (q : Quadrilateral) :
  q.A = ((q.a + q.c) / 2) * ((q.b + q.d) / 2) ↔ is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_area_equality_iff_rectangle_l3420_342062


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3420_342073

theorem no_function_satisfies_inequality :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y z : ℝ), f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3420_342073


namespace NUMINAMATH_CALUDE_ellipse_equation_l3420_342043

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0

/-- The right focus of the ellipse -/
def right_focus (e : Ellipse) : ℝ × ℝ := (3, 0)

/-- The midpoint of the line segment AB -/
def midpoint_AB : ℝ × ℝ := (1, -1)

/-- Theorem: Given an ellipse with the specified properties, its equation is x²/18 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) 
  (h2 : right_focus e = (3, 0))
  (h3 : midpoint_AB = (1, -1)) :
  ∃ (x y : ℝ), x^2 / 18 + y^2 / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3420_342043


namespace NUMINAMATH_CALUDE_smallest_divisor_perfect_cube_l3420_342070

theorem smallest_divisor_perfect_cube : ∃! n : ℕ, 
  n > 0 ∧ 
  n ∣ 34560 ∧ 
  (∃ m : ℕ, 34560 / n = m^3) ∧
  (∀ k : ℕ, k > 0 → k ∣ 34560 → (∃ l : ℕ, 34560 / k = l^3) → k ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_perfect_cube_l3420_342070


namespace NUMINAMATH_CALUDE_tan_beta_plus_pi_third_l3420_342006

theorem tan_beta_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - π/3) = 1/3) : 
  Real.tan (β + π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_plus_pi_third_l3420_342006


namespace NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_intersections_l3420_342004

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola y = x^2 -/
def Parabola := {p : Point | p.y = p.x^2}

/-- Checks if a point is on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if a point is on the parabola y = x^2 -/
def onParabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if two curves have coinciding tangents at a point -/
def coincidingTangents (p : Point) : Prop := sorry

/-- The main theorem -/
theorem tangents_not_necessarily_coincide_at_both_intersections
  (c : Circle) (A B : Point) :
  onCircle A c → onCircle B c →
  onParabola A → onParabola B →
  A ≠ B →
  coincidingTangents A →
  ¬ ∀ (c : Circle) (A B : Point),
    onCircle A c → onCircle B c →
    onParabola A → onParabola B →
    A ≠ B →
    coincidingTangents A →
    coincidingTangents B :=
by sorry

end NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_intersections_l3420_342004


namespace NUMINAMATH_CALUDE_carries_work_hours_l3420_342087

/-- Proves that Carrie works 35 hours per week given the problem conditions -/
theorem carries_work_hours :
  let hourly_rate : ℕ := 8
  let weeks_worked : ℕ := 4
  let bike_cost : ℕ := 400
  let money_left : ℕ := 720
  let total_earned : ℕ := bike_cost + money_left
  ∃ (hours_per_week : ℕ),
    hours_per_week * hourly_rate * weeks_worked = total_earned ∧
    hours_per_week = 35
  := by sorry

end NUMINAMATH_CALUDE_carries_work_hours_l3420_342087


namespace NUMINAMATH_CALUDE_art_students_l3420_342049

/-- Proves that the number of students taking art is 20 -/
theorem art_students (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 30)
  (h3 : both = 10)
  (h4 : neither = 460) :
  total - neither - (music - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_art_students_l3420_342049


namespace NUMINAMATH_CALUDE_compare_roots_l3420_342005

theorem compare_roots : 
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧ 
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧ 
  (16 : ℝ) ^ (1/16) > (27 : ℝ) ^ (1/27) := by
  sorry

#check compare_roots

end NUMINAMATH_CALUDE_compare_roots_l3420_342005


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3420_342088

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3420_342088


namespace NUMINAMATH_CALUDE_solve_for_m_l3420_342008

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n^2 - m

-- State the theorem
theorem solve_for_m :
  (∀ m n, customOp m n = n^2 - m) →
  (∃ m, customOp m 3 = 3) →
  (∃ m, m = 6) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_m_l3420_342008


namespace NUMINAMATH_CALUDE_min_ab_in_triangle_l3420_342060

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 2c cos B = 2a + b and the area S = √3 c, then ab ≥ 48. -/
theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 * c →
  ab ≥ 48 := by
sorry

end NUMINAMATH_CALUDE_min_ab_in_triangle_l3420_342060


namespace NUMINAMATH_CALUDE_divisor_sum_totient_inequality_divisor_sum_totient_equality_l3420_342044

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_totient_inequality (n : ℕ) :
  1 / (phi n : ℝ) + 1 / (sigma n : ℝ) ≥ 2 / n :=
sorry

/-- Characterization of the equality case -/
theorem divisor_sum_totient_equality (n : ℕ) :
  (1 / (phi n : ℝ) + 1 / (sigma n : ℝ) = 2 / n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_totient_inequality_divisor_sum_totient_equality_l3420_342044


namespace NUMINAMATH_CALUDE_disjoint_quadratic_sets_l3420_342096

theorem disjoint_quadratic_sets (A B : ℤ) : 
  ∃ C : ℤ, (∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C) := by
  sorry

end NUMINAMATH_CALUDE_disjoint_quadratic_sets_l3420_342096


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_29_l3420_342094

theorem sum_of_divisors_of_29 (h : Nat.Prime 29) : 
  (Finset.filter (· ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_29_l3420_342094


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3420_342058

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 3, -1; 1, -2, 5; 0, 6, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![1, 0, 4; 3, 2, -1; 0, 4, -2]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![11, 2, 7; -5, 16, -4; 18, 16, -8]

theorem matrix_multiplication_result : A * B = C := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3420_342058


namespace NUMINAMATH_CALUDE_phil_cards_l3420_342064

/-- Calculates the number of baseball cards remaining after buying for a year and losing half. -/
def remaining_cards (cards_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  (cards_per_week * weeks_per_year) / 2

/-- Theorem stating that buying 20 cards each week for 52 weeks and losing half results in 520 cards. -/
theorem phil_cards : remaining_cards 20 52 = 520 := by
  sorry

end NUMINAMATH_CALUDE_phil_cards_l3420_342064


namespace NUMINAMATH_CALUDE_job_land_theorem_l3420_342072

/-- Represents the total land owned by Job in hectares -/
def total_land : ℕ := 150

/-- Represents the land occupied by house and farm machinery in hectares -/
def house_and_machinery : ℕ := 25

/-- Represents the land reserved for future expansion in hectares -/
def future_expansion : ℕ := 15

/-- Represents the land dedicated to rearing cattle in hectares -/
def cattle_land : ℕ := 40

/-- Represents the land used for crop production in hectares -/
def crop_land : ℕ := 70

/-- Theorem stating that the total land is equal to the sum of all land uses -/
theorem job_land_theorem : 
  total_land = house_and_machinery + future_expansion + cattle_land + crop_land := by
  sorry

end NUMINAMATH_CALUDE_job_land_theorem_l3420_342072


namespace NUMINAMATH_CALUDE_seventh_term_of_sequence_l3420_342038

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem seventh_term_of_sequence (a₁ q : ℝ) (h₁ : a₁ = 3) (h₂ : q = Real.sqrt 2) :
  geometric_sequence a₁ q 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_sequence_l3420_342038


namespace NUMINAMATH_CALUDE_initial_friends_count_l3420_342054

def car_cost : ℕ := 1700
def car_wash_earnings : ℕ := 500
def cost_increase : ℕ := 40

theorem initial_friends_count : 
  ∃ (F : ℕ), 
    F > 0 ∧
    (car_cost - car_wash_earnings) / F + cost_increase = 
    (car_cost - car_wash_earnings) / (F - 1) ∧
    F = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_friends_count_l3420_342054


namespace NUMINAMATH_CALUDE_average_marks_proof_l3420_342066

theorem average_marks_proof (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 20) :
  (M + C) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l3420_342066


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3420_342093

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3420_342093


namespace NUMINAMATH_CALUDE_area_between_circles_l3420_342048

theorem area_between_circles (R r : ℝ) (h : ℝ) : 
  R = 10 → h = 16 → r^2 = R^2 - (h/2)^2 → (R^2 - r^2) * π = 64 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l3420_342048


namespace NUMINAMATH_CALUDE_simplify_exponential_expression_l3420_342015

theorem simplify_exponential_expression (a : ℝ) (h : a ≠ 0) :
  (a^9 * a^15) / a^3 = a^21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_expression_l3420_342015


namespace NUMINAMATH_CALUDE_meeting_calculation_correct_l3420_342041

/-- Represents the meeting of a pedestrian and cyclist --/
structure Meeting where
  time : ℝ  -- Time since start (in hours)
  distance : ℝ  -- Distance from the city (in km)

/-- Calculates the meeting point of a pedestrian and cyclist --/
def calculate_meeting (city_distance : ℝ) (pedestrian_speed : ℝ) (cyclist_speed : ℝ) (cyclist_rest : ℝ) : Meeting :=
  { time := 1.25,  -- 9:15 AM is 1.25 hours after 8:00 AM
    distance := 4.5 }

/-- Theorem stating the correctness of the meeting calculation --/
theorem meeting_calculation_correct 
  (city_distance : ℝ) 
  (pedestrian_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (cyclist_rest : ℝ)
  (h1 : city_distance = 12)
  (h2 : pedestrian_speed = 6)
  (h3 : cyclist_speed = 18)
  (h4 : cyclist_rest = 1/3)  -- 20 minutes is 1/3 of an hour
  : 
  let meeting := calculate_meeting city_distance pedestrian_speed cyclist_speed cyclist_rest
  meeting.time = 1.25 ∧ meeting.distance = 4.5 := by
  sorry

#check meeting_calculation_correct

end NUMINAMATH_CALUDE_meeting_calculation_correct_l3420_342041


namespace NUMINAMATH_CALUDE_divided_volumes_theorem_l3420_342040

/-- Regular triangular prism with base side length 2√14 -/
structure RegularTriangularPrism where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2 * Real.sqrt 14

/-- Plane dividing the prism -/
structure DividingPlane where
  prism : RegularTriangularPrism
  parallel_to_diagonal : Bool
  passes_through_vertex : Bool
  passes_through_center : Bool
  cross_section_area : ℝ
  cross_section_area_eq : cross_section_area = 21

/-- Volumes of the parts created by the dividing plane -/
def divided_volumes (p : RegularTriangularPrism) (d : DividingPlane) : (ℝ × ℝ) := sorry

/-- Theorem stating the volumes of the divided parts -/
theorem divided_volumes_theorem (p : RegularTriangularPrism) (d : DividingPlane) :
  d.prism = p → divided_volumes p d = (112/3, 154/3) := by sorry

end NUMINAMATH_CALUDE_divided_volumes_theorem_l3420_342040


namespace NUMINAMATH_CALUDE_determinant_scaling_l3420_342007

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 63 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3420_342007


namespace NUMINAMATH_CALUDE_expression_equals_two_l3420_342032

theorem expression_equals_two : 
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3420_342032


namespace NUMINAMATH_CALUDE_jeremys_songs_l3420_342068

theorem jeremys_songs (songs_yesterday songs_today total_songs : ℕ) : 
  songs_yesterday < songs_today →
  songs_yesterday = 9 →
  total_songs = 23 →
  songs_yesterday + songs_today = total_songs →
  songs_today = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremys_songs_l3420_342068


namespace NUMINAMATH_CALUDE_product_of_valid_m_l3420_342056

theorem product_of_valid_m : ∃ (S : Finset ℤ), 
  (∀ m ∈ S, m ≥ 1 ∧ 
    ∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) ∧ 
  (∀ m : ℤ, m ≥ 1 → 
    (∃ y : ℤ, y ≠ 2 ∧ m * y / (y - 2) + 1 = -3 * y / (2 - y)) → 
    m ∈ S) ∧
  S.prod id = 4 :=
sorry

end NUMINAMATH_CALUDE_product_of_valid_m_l3420_342056


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3420_342090

theorem simplify_and_rationalize :
  (Real.sqrt 2 / Real.sqrt 3) * (Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 6 / Real.sqrt 7) = 4 * Real.sqrt 35 / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3420_342090


namespace NUMINAMATH_CALUDE_garbage_classification_test_l3420_342084

theorem garbage_classification_test (p_idea : ℝ) (p_no_idea : ℝ) (p_B : ℝ) :
  p_idea = 2/3 →
  p_no_idea = 1/4 →
  p_B = 0.6 →
  let E_A := (3/4 * p_idea + 1/4 * p_no_idea) * 2
  let E_B := p_B * 2
  E_B > E_A :=
by sorry

end NUMINAMATH_CALUDE_garbage_classification_test_l3420_342084


namespace NUMINAMATH_CALUDE_ceiling_lights_difference_l3420_342085

theorem ceiling_lights_difference (medium large small : ℕ) : 
  medium = 12 →
  large = 2 * medium →
  small + 2 * medium + 3 * large = 118 →
  small - medium = 10 := by
sorry

end NUMINAMATH_CALUDE_ceiling_lights_difference_l3420_342085


namespace NUMINAMATH_CALUDE_smallest_regular_polygon_sides_l3420_342000

theorem smallest_regular_polygon_sides (n : ℕ) : n > 0 → (∃ k : ℕ, k > 0 ∧ 360 * k / (2 * n) = 28) → n ≥ 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_regular_polygon_sides_l3420_342000


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3420_342022

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 5 → x > 4) ∧
  (∃ x : ℝ, x > 4 ∧ x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3420_342022


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l3420_342029

-- Define the curve
def curve (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value (a : ℝ) :
  curve a (-1) = a + 2 →
  curve_derivative a (-1) = 8 →
  a = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l3420_342029


namespace NUMINAMATH_CALUDE_simplify_expression_l3420_342026

theorem simplify_expression (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3420_342026


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l3420_342079

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5)

theorem smallest_number_with_properties : 
  ∀ n : ℕ, n < 2019 → 
    ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by
  sorry

#check smallest_number_with_properties

end NUMINAMATH_CALUDE_smallest_number_with_properties_l3420_342079


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3420_342018

-- Define the circle equation
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 8*x + 6*y

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (1, 1)
def point_C : ℝ × ℝ := (4, 2)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 = 0 ∧
  circle_equation point_B.1 point_B.2 = 0 ∧
  circle_equation point_C.1 point_C.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3420_342018


namespace NUMINAMATH_CALUDE_max_a_value_l3420_342098

theorem max_a_value (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 17) 
  (h4 : (3/4) * x = (5/6) * y + a) (h5 : a > 0) : a < 51/4 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l3420_342098


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3420_342074

theorem triangle_perimeter_range (a b c : ℝ) : 
  a = 2 → b = 7 → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  14 < a + b + c ∧ a + b + c < 18 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3420_342074


namespace NUMINAMATH_CALUDE_queen_spade_probability_l3420_342081

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents a Queen card -/
def is_queen (card : Nat × Nat) : Prop := card.1 = 12

/-- Represents a Spade card -/
def is_spade (card : Nat × Nat) : Prop := card.2 = 3

/-- The probability of drawing a Queen as the first card and a Spade as the second card -/
def queen_spade_prob (d : Deck) : ℚ :=
  18 / 221

theorem queen_spade_probability (d : Deck) :
  queen_spade_prob d = 18 / 221 :=
sorry

end NUMINAMATH_CALUDE_queen_spade_probability_l3420_342081


namespace NUMINAMATH_CALUDE_cone_sphere_volume_difference_l3420_342013

/-- Given an equilateral cone with an inscribed sphere, prove that the difference in volume
    between the cone and the sphere is (10/3) * √(2/π) dm³ when the surface area of the cone
    is 10 dm² more than the surface area of the sphere. -/
theorem cone_sphere_volume_difference (R : ℝ) (h : R > 0) :
  let r := R / Real.sqrt 3
  let cone_surface_area := 3 * Real.pi * R^2
  let sphere_surface_area := 4 * Real.pi * r^2
  let cone_volume := (Real.pi * Real.sqrt 3 / 3) * R^3
  let sphere_volume := (4 * Real.pi / 3) * r^3
  cone_surface_area = sphere_surface_area + 10 →
  cone_volume - sphere_volume = (10 / 3) * Real.sqrt (2 / Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_difference_l3420_342013


namespace NUMINAMATH_CALUDE_expression_evaluation_l3420_342028

theorem expression_evaluation :
  (15 + 12)^2 - (12^2 + 15^2 + 6 * 15 * 12) = -720 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3420_342028


namespace NUMINAMATH_CALUDE_power_product_evaluation_l3420_342063

theorem power_product_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l3420_342063


namespace NUMINAMATH_CALUDE_tim_zoo_cost_l3420_342091

/-- The total cost of Tim's animals for his zoo -/
def total_cost (num_goats : ℕ) (goat_cost : ℚ) : ℚ :=
  let num_llamas := 2 * num_goats
  let llama_cost := goat_cost * (1 + 1/2)
  num_goats * goat_cost + num_llamas * llama_cost

/-- Theorem stating that Tim's total cost for animals is $4800 -/
theorem tim_zoo_cost : total_cost 3 400 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_tim_zoo_cost_l3420_342091


namespace NUMINAMATH_CALUDE_distinct_strings_equal_fibonacci_l3420_342037

/-- Represents the possible operations on a string --/
inductive Operation
  | replaceH
  | replaceMM
  | replaceT

/-- Defines a valid string after operations --/
def ValidString : Type := List Char

/-- Applies an operation to a valid string --/
def applyOperation (s : ValidString) (op : Operation) : ValidString :=
  sorry

/-- Counts the number of distinct strings after n operations --/
def countDistinctStrings (n : Nat) : Nat :=
  sorry

/-- Computes the nth Fibonacci number (starting with F(1) = 2, F(2) = 3) --/
def fibonacci (n : Nat) : Nat :=
  sorry

/-- The main theorem: number of distinct strings after 10 operations equals 10th Fibonacci number --/
theorem distinct_strings_equal_fibonacci :
  countDistinctStrings 10 = fibonacci 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_strings_equal_fibonacci_l3420_342037


namespace NUMINAMATH_CALUDE_complex_magnitude_l3420_342012

theorem complex_magnitude (z : ℂ) (h : z + (z - 1) * Complex.I = 3) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3420_342012


namespace NUMINAMATH_CALUDE_no_multiples_of_five_end_in_two_l3420_342065

theorem no_multiples_of_five_end_in_two :
  {n : ℕ | n > 0 ∧ n < 500 ∧ n % 5 = 0 ∧ n % 10 = 2} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_multiples_of_five_end_in_two_l3420_342065


namespace NUMINAMATH_CALUDE_button_sequence_l3420_342023

theorem button_sequence (a : Fin 6 → ℕ) (h1 : a 0 = 1)
    (h2 : a 1 = 3) (h4 : a 3 = 27) (h5 : a 4 = 81) (h6 : a 5 = 243)
    (h_ratio : ∀ i : Fin 5, a (i + 1) = 3 * a i) : a 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_button_sequence_l3420_342023


namespace NUMINAMATH_CALUDE_snow_cover_probabilities_l3420_342027

theorem snow_cover_probabilities (p : ℝ) (h : p = 0.2) :
  let q := 1 - p
  (q^2 = 0.64) ∧ (1 - q^2 = 0.36) := by
  sorry

end NUMINAMATH_CALUDE_snow_cover_probabilities_l3420_342027


namespace NUMINAMATH_CALUDE_system_solution_l3420_342092

theorem system_solution (x y : ℝ) : 
  (x - y = 2 ∧ 3 * x + y = 4) ↔ (x = 1.5 ∧ y = -0.5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3420_342092


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_plus_poly_l3420_342034

/-- A nonzero polynomial with integer coefficients -/
def nonzero_int_poly (P : ℕ → ℤ) : Prop :=
  ∃ n, P n ≠ 0 ∧ ∀ m, ∃ k : ℤ, P m = k

theorem infinitely_many_primes_dividing_2_pow_plus_poly 
  (P : ℕ → ℤ) (h : nonzero_int_poly P) :
  ∀ N : ℕ, ∃ q : ℕ, q > N ∧ Nat.Prime q ∧ 
    ∃ n : ℕ, (q : ℤ) ∣ (2^n : ℤ) + P n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_2_pow_plus_poly_l3420_342034


namespace NUMINAMATH_CALUDE_rectangle_area_l3420_342024

/-- Given a rectangle with perimeter 280 meters and length-to-width ratio of 5:2, its area is 4000 square meters. -/
theorem rectangle_area (L W : ℝ) (h1 : 2*L + 2*W = 280) (h2 : L / W = 5 / 2) : L * W = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3420_342024


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3420_342057

/-- Tangent line to a circle -/
theorem tangent_line_to_circle (r x_0 y_0 : ℝ) (h : x_0^2 + y_0^2 = r^2) :
  ∀ x y : ℝ, (x^2 + y^2 = r^2) → ((x - x_0)^2 + (y - y_0)^2 = 0 ∨ x_0*x + y_0*y = r^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3420_342057


namespace NUMINAMATH_CALUDE_function_range_l3420_342099

theorem function_range (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * (4 - m) * x + 1 > 0) ∨ (m * x > 0)) → 
  (m > 0 ∧ m < 8) := by
sorry

end NUMINAMATH_CALUDE_function_range_l3420_342099


namespace NUMINAMATH_CALUDE_expression_value_l3420_342052

theorem expression_value (a b c : ℤ) (ha : a = 17) (hb : b = 21) (hc : c = 5) :
  (a - (b - c)) - ((a - b) - c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3420_342052


namespace NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l3420_342047

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧ 
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ c : ℕ, m = c^2) ∧ (∃ d : ℕ, m = d^3) → m ≥ n) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l3420_342047


namespace NUMINAMATH_CALUDE_car_speed_problem_l3420_342082

/-- Proves that given the conditions of the car problem, the speed of Car X is approximately 33.87 mph -/
theorem car_speed_problem (speed_y : ℝ) (time_diff : ℝ) (distance_x : ℝ) :
  speed_y = 42 →
  time_diff = 72 / 60 →
  distance_x = 210 →
  ∃ (speed_x : ℝ), 
    speed_x > 0 ∧ 
    speed_x * (distance_x / speed_y + time_diff) = distance_x ∧ 
    (abs (speed_x - 33.87) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3420_342082


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3420_342059

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + 4*x - 5 > 0} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3420_342059


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3420_342030

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3420_342030


namespace NUMINAMATH_CALUDE_common_chord_length_is_sqrt55_div_5_l3420_342025

noncomputable section

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

def circle_C2_center : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

def circle_C2_radius : ℝ := 1

-- Define the length of the common chord
def common_chord_length : ℝ := Real.sqrt 55 / 5

-- Theorem statement
theorem common_chord_length_is_sqrt55_div_5 :
  ∃ (A B : ℝ × ℝ),
    (circle_C1 A.1 A.2) ∧
    (circle_C1 B.1 B.2) ∧
    ((A.1 - circle_C2_center.1)^2 + (A.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    ((B.1 - circle_C2_center.1)^2 + (B.2 - circle_C2_center.2)^2 = circle_C2_radius^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_sqrt55_div_5_l3420_342025


namespace NUMINAMATH_CALUDE_scientific_notation_4212000_l3420_342039

theorem scientific_notation_4212000 :
  ∃ (a : ℝ) (n : ℤ), 
    4212000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ a ∧ 
    a < 10 ∧ 
    a = 4.212 ∧ 
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4212000_l3420_342039
