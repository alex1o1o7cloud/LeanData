import Mathlib

namespace NUMINAMATH_CALUDE_fixed_points_m_range_l1180_118059

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * m * x + 4

-- Theorem for fixed points
theorem fixed_points (m : ℝ) : f m 0 = 4 ∧ f m 6 = 4 := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) (a b c : ℝ) 
  (h1 : f m 1 = a) (h2 : f m 3 = b) (h3 : f m 4 = c) (h4 : a * b * c < 0) : 
  (4/9 < m ∧ m < 1/2) ∨ m > 4/5 := by sorry

end NUMINAMATH_CALUDE_fixed_points_m_range_l1180_118059


namespace NUMINAMATH_CALUDE_halloween_goodie_bags_l1180_118037

theorem halloween_goodie_bags (vampire_students pumpkin_students : ℕ)
  (pack_size pack_cost individual_cost total_cost : ℕ) :
  vampire_students = 11 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_cost = 3 →
  individual_cost = 1 →
  total_cost = 17 →
  vampire_students + pumpkin_students = 25 :=
by sorry

end NUMINAMATH_CALUDE_halloween_goodie_bags_l1180_118037


namespace NUMINAMATH_CALUDE_final_salary_correct_l1180_118009

/-- Calculates the final salary after a series of changes -/
def calculate_final_salary (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) : ℝ :=
  let salary_after_raise := initial_salary * (1 + raise_percentage)
  let salary_after_cut := salary_after_raise * (1 - cut_percentage)
  salary_after_cut - deduction

/-- Theorem stating that the final salary matches the expected value -/
theorem final_salary_correct (initial_salary : ℝ) (raise_percentage : ℝ) (cut_percentage : ℝ) (deduction : ℝ) 
    (h1 : initial_salary = 3000)
    (h2 : raise_percentage = 0.1)
    (h3 : cut_percentage = 0.15)
    (h4 : deduction = 100) :
  calculate_final_salary initial_salary raise_percentage cut_percentage deduction = 2705 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_correct_l1180_118009


namespace NUMINAMATH_CALUDE_simplify_expression_l1180_118004

variable (y : ℝ)

theorem simplify_expression :
  3 * y - 5 * y^2 + 7 - (6 - 3 * y + 5 * y^2 - 2 * y^3) = 2 * y^3 - 10 * y^2 + 6 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1180_118004


namespace NUMINAMATH_CALUDE_slope_of_l₃_l1180_118083

-- Define the lines and points
def l₁ : Set (ℝ × ℝ) := {(x, y) | 5 * x - 3 * y = 2}
def l₂ : Set (ℝ × ℝ) := {(x, y) | y = 2}
def A : ℝ × ℝ := (2, -2)

-- Define the existence of point B
def B_exists : Prop := ∃ B : ℝ × ℝ, B ∈ l₁ ∧ B ∈ l₂

-- Define the existence of point C and line l₃
def C_and_l₃_exist : Prop := ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
  C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) > 0)

-- Define the area of triangle ABC
def area_ABC : ℝ := 5

-- Theorem statement
theorem slope_of_l₃ (h₁ : A ∈ l₁) (h₂ : B_exists) (h₃ : C_and_l₃_exist) (h₄ : area_ABC = 5) :
  ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
    C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
    (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 20 / 9) :=
sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l1180_118083


namespace NUMINAMATH_CALUDE_roller_coaster_height_l1180_118070

/-- The required height to ride the roller coaster given Alex's current height,
    normal growth rate, additional growth rate from hanging upside down,
    and the required hanging time. -/
theorem roller_coaster_height
  (current_height : ℝ)
  (normal_growth_rate : ℝ)
  (upside_down_growth_rate : ℝ)
  (hanging_time : ℝ)
  (months_per_year : ℕ)
  (h1 : current_height = 48)
  (h2 : normal_growth_rate = 1 / 3)
  (h3 : upside_down_growth_rate = 1 / 12)
  (h4 : hanging_time = 2)
  (h5 : months_per_year = 12) :
  current_height +
  normal_growth_rate * months_per_year +
  upside_down_growth_rate * hanging_time * months_per_year = 54 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_height_l1180_118070


namespace NUMINAMATH_CALUDE_unique_integer_l1180_118051

def is_valid_integer (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 14 ∧
    b + c = 9 ∧
    a - d = 1 ∧
    n % 11 = 0

theorem unique_integer : ∃! n : ℕ, is_valid_integer n ∧ n = 3542 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_l1180_118051


namespace NUMINAMATH_CALUDE_mistaken_calculation_l1180_118042

theorem mistaken_calculation (x : ℕ) : 
  (x / 16 = 8) → (x % 16 = 4) → (x * 16 + 8 = 2120) :=
by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l1180_118042


namespace NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l1180_118090

theorem find_integers_with_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 3972 ∧ 
    Nat.lcm a b = 985928 ∧ 
    a = 1964 ∧ 
    b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l1180_118090


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l1180_118001

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis) ∧ ¬Andrew ∧ ¬Gena :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l1180_118001


namespace NUMINAMATH_CALUDE_ordering_of_powers_l1180_118045

theorem ordering_of_powers : 5^(1/5) > 0.5^(1/5) ∧ 0.5^(1/5) > 0.5^2 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l1180_118045


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l1180_118021

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1 / 4) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l1180_118021


namespace NUMINAMATH_CALUDE_special_list_median_l1180_118012

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem special_list_median :
  median special_list = 71 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l1180_118012


namespace NUMINAMATH_CALUDE_highest_sample_number_l1180_118064

/-- Given a systematic sample from a population, calculate the highest number in the sample. -/
theorem highest_sample_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_sample : Nat)
  (h1 : total_students = 54)
  (h2 : sample_size = 6)
  (h3 : first_sample = 5)
  (h4 : sample_size > 0)
  (h5 : total_students ≥ sample_size)
  : first_sample + (sample_size - 1) * (total_students / sample_size) = 50 := by
  sorry

#check highest_sample_number

end NUMINAMATH_CALUDE_highest_sample_number_l1180_118064


namespace NUMINAMATH_CALUDE_new_songs_added_l1180_118077

def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

theorem new_songs_added : 
  final_songs - (initial_songs - deleted_songs) = 20 := by sorry

end NUMINAMATH_CALUDE_new_songs_added_l1180_118077


namespace NUMINAMATH_CALUDE_sphere_equation_l1180_118041

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a sphere in 3D space -/
def Sphere (center : Point3D) (radius : ℝ) : Set Point3D :=
  {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2}

/-- Theorem: The equation (x - x₀)² + (y - y₀)² + (z - z₀)² = r² represents a sphere
    with center (x₀, y₀, z₀) and radius r in a three-dimensional Cartesian coordinate system -/
theorem sphere_equation (center : Point3D) (radius : ℝ) :
  Sphere center radius = {p : Point3D | (p.x - center.x)^2 + (p.y - center.y)^2 + (p.z - center.z)^2 = radius^2} := by
  sorry

end NUMINAMATH_CALUDE_sphere_equation_l1180_118041


namespace NUMINAMATH_CALUDE_min_representatives_per_table_l1180_118031

/-- Represents the number of representatives for each country -/
structure Representatives where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The condition that country ratios are satisfied -/
def satisfies_ratios (r : Representatives) : Prop :=
  r.A = 2 * r.B ∧ r.A = 3 * r.C ∧ r.A = 4 * r.D

/-- The condition that each country is outnumbered by others at a table -/
def is_outnumbered (r : Representatives) (total : ℕ) : Prop :=
  r.A < r.B + r.C + r.D ∧
  r.B < r.A + r.C + r.D ∧
  r.C < r.A + r.B + r.D ∧
  r.D < r.A + r.B + r.C

/-- The main theorem stating the minimum number of representatives per table -/
theorem min_representatives_per_table (r : Representatives) 
  (h_ratios : satisfies_ratios r) : 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) → 
  (∃ (n : ℕ), n > 0 ∧ is_outnumbered r n ∧ 
    ∀ (m : ℕ), m > 0 ∧ is_outnumbered r m → n ≤ m) ∧ n = 25 :=
sorry

end NUMINAMATH_CALUDE_min_representatives_per_table_l1180_118031


namespace NUMINAMATH_CALUDE_domain_of_f_l1180_118010

noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^3 - 5*x^2 + 8*x - 4)

theorem domain_of_f :
  Set.range f = {x : ℝ | x ∈ (Set.Iio 1) ∪ (Set.Ioo 1 2) ∪ (Set.Ioo 2 4) ∪ (Set.Ioi 4)} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l1180_118010


namespace NUMINAMATH_CALUDE_rogue_trader_goods_value_l1180_118084

def base7ToBase10 (n : ℕ) : ℕ := sorry

def spiceValue : ℕ := 5213
def metalValue : ℕ := 1653
def fruitValue : ℕ := 202

theorem rogue_trader_goods_value :
  base7ToBase10 spiceValue + base7ToBase10 metalValue + base7ToBase10 fruitValue = 2598 := by
  sorry

end NUMINAMATH_CALUDE_rogue_trader_goods_value_l1180_118084


namespace NUMINAMATH_CALUDE_scientific_notation_21600_l1180_118034

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_21600 :
  toScientificNotation 21600 = ScientificNotation.mk 2.16 4 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_21600_l1180_118034


namespace NUMINAMATH_CALUDE_subtraction_result_l1180_118036

theorem subtraction_result : (3.75 : ℝ) - 1.4 = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1180_118036


namespace NUMINAMATH_CALUDE_polynomial_division_identity_l1180_118056

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^6 - 5*x^4 + 3*x^3 - 7*x^2 + 2*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def q (x : ℝ) : ℝ := x^5 + 3*x^4 + 4*x^3 + 15*x^2 + 38*x + 116

/-- The remainder -/
def r : ℝ := 340

/-- Theorem stating the polynomial division identity -/
theorem polynomial_division_identity : 
  ∀ x : ℝ, f x = g x * q x + r := by sorry

end NUMINAMATH_CALUDE_polynomial_division_identity_l1180_118056


namespace NUMINAMATH_CALUDE_valid_draw_count_l1180_118000

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def cards_drawn : ℕ := 3

def valid_draw (total : ℕ) (per_color : ℕ) (drawn : ℕ) : ℕ :=
  Nat.choose total drawn - 
  4 * Nat.choose per_color drawn - 
  Nat.choose per_color 2 * Nat.choose (total - per_color) 1

theorem valid_draw_count :
  valid_draw total_cards cards_per_color cards_drawn = 472 := by
  sorry

end NUMINAMATH_CALUDE_valid_draw_count_l1180_118000


namespace NUMINAMATH_CALUDE_distance_home_to_school_l1180_118006

/-- Represents Johnny's journey to and from school -/
structure JourneySegment where
  speed : ℝ
  time : ℝ
  distance : ℝ
  (distance_eq : distance = speed * time)

/-- Represents Johnny's complete journey -/
structure Journey where
  jog : JourneySegment
  bike : JourneySegment
  bus : JourneySegment

/-- The journey satisfies the given conditions -/
def journey_conditions (j : Journey) : Prop :=
  j.jog.speed = 5 ∧
  j.bike.speed = 10 ∧
  j.bus.speed = 30 ∧
  j.jog.time = 1 ∧
  j.bike.time = 1 ∧
  j.bus.time = 1

/-- The theorem stating the distance from home to school -/
theorem distance_home_to_school (j : Journey) 
  (h : journey_conditions j) : 
  j.bus.distance - j.bike.distance = 20 := by
  sorry


end NUMINAMATH_CALUDE_distance_home_to_school_l1180_118006


namespace NUMINAMATH_CALUDE_sum_of_divisors_77_and_not_perfect_l1180_118023

def sum_of_divisors (n : ℕ) : ℕ := sorry

def is_perfect_number (n : ℕ) : Prop :=
  sum_of_divisors n = 2 * n

theorem sum_of_divisors_77_and_not_perfect :
  sum_of_divisors 77 = 96 ∧ ¬(is_perfect_number 77) := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_77_and_not_perfect_l1180_118023


namespace NUMINAMATH_CALUDE_prime_product_digital_sum_difference_l1180_118027

def digital_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digital_sum (n / 10)

theorem prime_product_digital_sum_difference 
  (p q r : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hpqr : p * q * r = 18 * 962) 
  (hdiff : p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ∃ (result : ℕ), digital_sum p + digital_sum q + digital_sum r - digital_sum (p * q * r) = result :=
sorry

end NUMINAMATH_CALUDE_prime_product_digital_sum_difference_l1180_118027


namespace NUMINAMATH_CALUDE_problem_solution_l1180_118065

theorem problem_solution (x : ℝ) : 
  (x - 9)^3 / (x + 4) = 27 → (x^2 - 12*x + 15) / (x - 2) = -20.1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1180_118065


namespace NUMINAMATH_CALUDE_weight_difference_l1180_118094

/-- Proves that Heather is 53.4 pounds lighter than Emily, Elizabeth, and George combined -/
theorem weight_difference (heather emily elizabeth george : ℝ) 
  (h1 : heather = 87.5)
  (h2 : emily = 45.3)
  (h3 : elizabeth = 38.7)
  (h4 : george = 56.9) :
  heather - (emily + elizabeth + george) = -53.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l1180_118094


namespace NUMINAMATH_CALUDE_distance_between_blue_lights_l1180_118028

/-- Represents the pattern of lights -/
inductive LightColor
| Blue
| Yellow

/-- Represents the recurring pattern of lights -/
def lightPattern : List LightColor :=
  [LightColor.Blue, LightColor.Blue, LightColor.Blue,
   LightColor.Yellow, LightColor.Yellow, LightColor.Yellow, LightColor.Yellow]

/-- The spacing between lights in inches -/
def lightSpacing : ℕ := 7

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- Calculates the position of the nth blue light in the sequence -/
def bluePosition (n : ℕ) : ℕ :=
  ((n - 1) / 3) * lightPattern.length + ((n - 1) % 3) + 1

/-- The main theorem to prove -/
theorem distance_between_blue_lights :
  (bluePosition 25 - bluePosition 4) * lightSpacing / inchesPerFoot = 28 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_blue_lights_l1180_118028


namespace NUMINAMATH_CALUDE_bird_increase_l1180_118082

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds : ℕ := 1300

/-- The number of fish-eater birds Cohen saw on the first day -/
def first_day_birds : ℕ := 300

/-- The decrease in the number of birds from the first day to the third day -/
def third_day_decrease : ℕ := 200

/-- Theorem stating the increase in the number of birds from the first to the second day -/
theorem bird_increase : 
  ∃ (second_day_birds third_day_birds : ℕ), 
    first_day_birds + second_day_birds + third_day_birds = total_birds ∧
    third_day_birds = first_day_birds - third_day_decrease ∧
    second_day_birds = first_day_birds + 600 :=
by sorry

end NUMINAMATH_CALUDE_bird_increase_l1180_118082


namespace NUMINAMATH_CALUDE_pebble_difference_l1180_118008

/-- Represents the number of pebbles thrown by each person -/
structure PebbleCount where
  candy : ℚ
  lance : ℚ
  sandy : ℚ

/-- The pebble throwing scenario -/
def pebble_scenario (p : PebbleCount) : Prop :=
  p.lance = p.candy + 10 ∧ 
  5 * p.candy = 2 * p.lance ∧
  4 * p.candy = 2 * p.sandy

theorem pebble_difference (p : PebbleCount) 
  (h : pebble_scenario p) : 
  p.lance + p.sandy - p.candy = 30 := by
  sorry

#check pebble_difference

end NUMINAMATH_CALUDE_pebble_difference_l1180_118008


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l1180_118013

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the sequence of selected students -/
def generate_sequence (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if a sequence is valid for the given systematic sample -/
def is_valid_sequence (s : SystematicSample) (seq : List Nat) : Prop :=
  seq.length = s.sample_size ∧
  seq.all (· ≤ s.total) ∧
  seq = generate_sequence s

theorem correct_systematic_sample :
  let s : SystematicSample := ⟨50, 5, 3, 10⟩
  is_valid_sequence s [3, 13, 23, 33, 43] := by
  sorry

#eval generate_sequence ⟨50, 5, 3, 10⟩

end NUMINAMATH_CALUDE_correct_systematic_sample_l1180_118013


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1180_118011

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1180_118011


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1180_118015

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

theorem repeating_decimal_subtraction :
  RepeatingDecimal 2 3 4 5 - RepeatingDecimal 6 7 8 9 - RepeatingDecimal 1 2 3 4 = -5678 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1180_118015


namespace NUMINAMATH_CALUDE_complement_of_63_degrees_l1180_118093

theorem complement_of_63_degrees :
  let angle : ℝ := 63
  let complement (x : ℝ) : ℝ := 90 - x
  complement angle = 27 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_63_degrees_l1180_118093


namespace NUMINAMATH_CALUDE_tan_function_property_l1180_118024

/-- Given a function f(x) = a * tan(b * x) where a and b are positive constants,
    if f has vertical asymptotes at x = ±π/4 and passes through (π/8, 3),
    then a * b = 6 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π/4 ∧ x ≠ -π/4 → ∃ y, y = a * Real.tan (b * x)) →
  a * Real.tan (b * π/8) = 3 →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l1180_118024


namespace NUMINAMATH_CALUDE_paper_stack_height_l1180_118098

/-- Given a stack of paper where 800 sheets are 4 cm thick, 
    prove that a 6 cm high stack would contain 1200 sheets. -/
theorem paper_stack_height (sheets : ℕ) (height : ℝ) : 
  (800 : ℝ) / 4 = sheets / height → sheets = 1200 ∧ height = 6 :=
by sorry

end NUMINAMATH_CALUDE_paper_stack_height_l1180_118098


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l1180_118017

def scores : List Nat := [91, 68, 58, 99, 82, 94, 88, 76, 79, 62, 87, 81, 65, 85, 89, 73, 77, 84, 59, 72]

def is_grade_B (score : Nat) : Bool :=
  85 ≤ score ∧ score ≤ 94

def count_grade_B (scores : List Nat) : Nat :=
  scores.filter is_grade_B |>.length

theorem percentage_of_B_grades :
  (count_grade_B scores : ℚ) / scores.length * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l1180_118017


namespace NUMINAMATH_CALUDE_franks_chips_purchase_franks_chips_purchase_correct_l1180_118026

theorem franks_chips_purchase (chocolate_bars : ℕ) (chocolate_price : ℕ) 
  (chip_price : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let chocolate_cost := chocolate_bars * chocolate_price
  let chips_cost := total_spent - chocolate_cost
  chips_cost / chip_price

#check franks_chips_purchase 5 2 3 20 4 = 2

theorem franks_chips_purchase_correct : franks_chips_purchase 5 2 3 20 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_franks_chips_purchase_franks_chips_purchase_correct_l1180_118026


namespace NUMINAMATH_CALUDE_tom_chocolate_boxes_l1180_118007

/-- The number of pieces Tom gave away -/
def pieces_given_away : ℕ := 8

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of pieces Tom still has -/
def pieces_remaining : ℕ := 18

/-- The number of boxes Tom bought initially -/
def boxes_bought : ℕ := 8

theorem tom_chocolate_boxes :
  boxes_bought * pieces_per_box = pieces_given_away + pieces_remaining :=
by sorry

end NUMINAMATH_CALUDE_tom_chocolate_boxes_l1180_118007


namespace NUMINAMATH_CALUDE_distinct_values_count_l1180_118062

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def distinct_fractions (S : Finset ℕ) : Finset ℚ :=
  (S.product S).filter (fun (a, b) => a ≠ b)
    |>.image (fun (a, b) => (a : ℚ) / b)

theorem distinct_values_count :
  (distinct_fractions S).card = 22 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_count_l1180_118062


namespace NUMINAMATH_CALUDE_bus_tour_sales_l1180_118068

/-- Given a bus tour with senior and regular tickets, calculate the total sales amount. -/
theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : senior_tickets = 24)
  (h5 : senior_tickets ≤ total_tickets) :
  senior_tickets * senior_price + (total_tickets - senior_tickets) * regular_price = 855 := by
  sorry


end NUMINAMATH_CALUDE_bus_tour_sales_l1180_118068


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_isosceles_triangle_l1180_118099

-- Define the triangle
def right_isosceles_triangle (leg : ℝ) (hypotenuse : ℝ) : Prop :=
  leg > 0 ∧ hypotenuse > 0 ∧ hypotenuse^2 = 2 * leg^2

-- Theorem statement
theorem hypotenuse_of_right_isosceles_triangle :
  ∀ (leg : ℝ) (hypotenuse : ℝ),
  right_isosceles_triangle leg hypotenuse →
  leg = 8 →
  hypotenuse = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_isosceles_triangle_l1180_118099


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1180_118039

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1180_118039


namespace NUMINAMATH_CALUDE_three_flower_purchase_options_l1180_118049

/-- Represents a flower purchase option -/
structure FlowerPurchase where
  carnations : Nat
  lilies : Nat

/-- The cost of a single carnation in yuan -/
def carnationCost : Nat := 2

/-- The cost of a single lily in yuan -/
def lilyCost : Nat := 3

/-- The total amount Xiaoming has to spend in yuan -/
def totalSpend : Nat := 20

/-- Predicate to check if a flower purchase is valid -/
def isValidPurchase (purchase : FlowerPurchase) : Prop :=
  carnationCost * purchase.carnations + lilyCost * purchase.lilies = totalSpend

/-- The theorem stating that there are exactly 3 valid flower purchase options -/
theorem three_flower_purchase_options :
  ∃ (options : List FlowerPurchase),
    (options.length = 3) ∧
    (∀ purchase ∈ options, isValidPurchase purchase) ∧
    (∀ purchase, isValidPurchase purchase → purchase ∈ options) :=
sorry

end NUMINAMATH_CALUDE_three_flower_purchase_options_l1180_118049


namespace NUMINAMATH_CALUDE_greatest_multiple_5_and_6_less_than_800_l1180_118060

theorem greatest_multiple_5_and_6_less_than_800 : 
  ∃ n : ℕ, n = 780 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 800 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 800 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_5_and_6_less_than_800_l1180_118060


namespace NUMINAMATH_CALUDE_income_ratio_l1180_118047

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def financialProblem (p1 p2 : Person) : Prop :=
  p1.income = 3500 ∧
  p1.savings = 1400 ∧
  p2.savings = 1400 ∧
  p1.expenditure * 2 = p2.expenditure * 3 ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) 
  (h : financialProblem p1 p2) : 
  p1.income * 4 = p2.income * 5 := by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l1180_118047


namespace NUMINAMATH_CALUDE_equation_C_violates_basic_properties_l1180_118055

-- Define the equations
def equation_A (a b c : ℝ) : Prop := (a / c = b / c) → (a = b)
def equation_B (a b : ℝ) : Prop := (-a = -b) → (2 - a = 2 - b)
def equation_C (a b c : ℝ) : Prop := (a * c = b * c) → (a = b)
def equation_D (a b m : ℝ) : Prop := ((m^2 + 1) * a = (m^2 + 1) * b) → (a = b)

-- Theorem statement
theorem equation_C_violates_basic_properties :
  (∃ a b c : ℝ, ¬(equation_C a b c)) ∧
  (∀ a b c : ℝ, c ≠ 0 → equation_A a b c) ∧
  (∀ a b : ℝ, equation_B a b) ∧
  (∀ a b m : ℝ, equation_D a b m) :=
by sorry

end NUMINAMATH_CALUDE_equation_C_violates_basic_properties_l1180_118055


namespace NUMINAMATH_CALUDE_linear_function_fixed_point_l1180_118085

theorem linear_function_fixed_point (k : ℝ) : (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_fixed_point_l1180_118085


namespace NUMINAMATH_CALUDE_new_tires_cost_calculation_l1180_118089

/-- The amount spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The total amount spent on car parts -/
def total_car_parts_cost : ℝ := 224.87

/-- The amount spent on new tires -/
def new_tires_cost : ℝ := total_car_parts_cost - speakers_cost

theorem new_tires_cost_calculation : 
  new_tires_cost = 106.33 := by sorry

end NUMINAMATH_CALUDE_new_tires_cost_calculation_l1180_118089


namespace NUMINAMATH_CALUDE_tiled_square_theorem_l1180_118096

/-- A square area tiled with identical square tiles -/
structure TiledSquare where
  /-- The number of tiles adjoining the four sides -/
  perimeter_tiles : ℕ
  /-- The total number of tiles in the square -/
  total_tiles : ℕ

/-- Theorem stating that a square area with 20 tiles adjoining its sides contains 36 tiles in total -/
theorem tiled_square_theorem (ts : TiledSquare) (h : ts.perimeter_tiles = 20) : ts.total_tiles = 36 := by
  sorry

end NUMINAMATH_CALUDE_tiled_square_theorem_l1180_118096


namespace NUMINAMATH_CALUDE_solve_homework_problem_l1180_118020

def homework_problem (total_problems : ℕ) (completed_at_stop1 : ℕ) (completed_at_stop2 : ℕ) (completed_at_stop3 : ℕ) : Prop :=
  let completed_on_bus := completed_at_stop1 + completed_at_stop2 + completed_at_stop3
  let remaining_problems := total_problems - completed_on_bus
  remaining_problems = 3

theorem solve_homework_problem :
  homework_problem 9 2 3 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_homework_problem_l1180_118020


namespace NUMINAMATH_CALUDE_book_distribution_count_l1180_118044

/-- The number of ways to distribute books between the library and checked-out status -/
def distribution_count (total : ℕ) (min_in_library : ℕ) (max_in_library : ℕ) : ℕ :=
  (max_in_library - min_in_library + 1)

/-- Theorem stating the number of ways to distribute 8 books with given constraints -/
theorem book_distribution_count :
  distribution_count 8 2 6 = 5 := by sorry

end NUMINAMATH_CALUDE_book_distribution_count_l1180_118044


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1180_118074

theorem sum_of_squares_zero_implies_sum (a b c d : ℝ) :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 →
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1180_118074


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1180_118095

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (Real.sqrt x)
  ∀ x : ℝ, (x > 0 ∧ f x = 16 / (9 - f x)) ↔ (x = 4096 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1180_118095


namespace NUMINAMATH_CALUDE_symmetric_function_is_exponential_l1180_118043

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the symmetry condition
def symmetric_to_log3 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → (f x = y ↔ log3 y = x)

-- Theorem statement
theorem symmetric_function_is_exponential (f : ℝ → ℝ) :
  symmetric_to_log3 f → (∀ x : ℝ, f x = 3^x) :=
sorry

end NUMINAMATH_CALUDE_symmetric_function_is_exponential_l1180_118043


namespace NUMINAMATH_CALUDE_square_sum_eq_double_product_iff_equal_l1180_118058

theorem square_sum_eq_double_product_iff_equal (a b : ℝ) :
  a^2 + b^2 = 2*a*b ↔ a = b := by
sorry

end NUMINAMATH_CALUDE_square_sum_eq_double_product_iff_equal_l1180_118058


namespace NUMINAMATH_CALUDE_total_time_circling_island_l1180_118022

/-- The time in minutes to navigate around the island once. -/
def time_per_round : ℕ := 30

/-- The number of rounds completed on Saturday. -/
def saturday_rounds : ℕ := 11

/-- The number of rounds completed on Sunday. -/
def sunday_rounds : ℕ := 15

/-- The total time spent circling the island over the weekend. -/
theorem total_time_circling_island : 
  (saturday_rounds + sunday_rounds) * time_per_round = 780 := by sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l1180_118022


namespace NUMINAMATH_CALUDE_v_2004_equals_1_l1180_118029

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for completeness

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n + 1)

-- Theorem statement
theorem v_2004_equals_1 : v 2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2004_equals_1_l1180_118029


namespace NUMINAMATH_CALUDE_condition_property_l1180_118040

theorem condition_property :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 1 ∨ y ≠ 4)) ∧
  (∃ x y : ℝ, (x ≠ 1 ∨ y ≠ 4) ∧ x + y = 5) := by
  sorry

end NUMINAMATH_CALUDE_condition_property_l1180_118040


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l1180_118035

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_negative_one 
  (a b c : ℝ) 
  (h : (4 * a + 2 * b) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l1180_118035


namespace NUMINAMATH_CALUDE_circle_symmetry_l1180_118003

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), circle_C x y ∧ line_l x y → symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1180_118003


namespace NUMINAMATH_CALUDE_road_breadth_from_fallen_tree_l1180_118092

/-- The breadth of a road when a tree falls across it -/
theorem road_breadth_from_fallen_tree (tree_height : ℝ) (break_height : ℝ) (road_breadth : ℝ) : 
  tree_height = 36 →
  break_height = 16 →
  (tree_height - break_height) ^ 2 = road_breadth ^ 2 + break_height ^ 2 →
  road_breadth = 12 := by
sorry

end NUMINAMATH_CALUDE_road_breadth_from_fallen_tree_l1180_118092


namespace NUMINAMATH_CALUDE_cookies_prepared_l1180_118071

theorem cookies_prepared (num_guests : ℕ) (cookies_per_guest : ℕ) : 
  num_guests = 10 → cookies_per_guest = 18 → num_guests * cookies_per_guest = 180 := by
  sorry

#check cookies_prepared

end NUMINAMATH_CALUDE_cookies_prepared_l1180_118071


namespace NUMINAMATH_CALUDE_prob_at_least_one_spade_or_ace_value_l1180_118050

/-- The number of cards in the deck -/
def deck_size : ℕ := 54

/-- The number of cards that are either spades or aces -/
def spade_or_ace_count : ℕ := 16

/-- The probability of drawing at least one spade or ace in two independent draws with replacement -/
def prob_at_least_one_spade_or_ace : ℚ :=
  1 - (1 - spade_or_ace_count / deck_size) ^ 2

theorem prob_at_least_one_spade_or_ace_value :
  prob_at_least_one_spade_or_ace = 368 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_spade_or_ace_value_l1180_118050


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1180_118030

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 14) 
  (h_a5 : a 5 = 5) : 
  common_difference a = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1180_118030


namespace NUMINAMATH_CALUDE_parallel_vector_sum_diff_l1180_118002

/-- Given two vectors a and b in ℝ², where a = (1, -1) and b = (t, 1),
    if a + b is parallel to a - b, then t = -1. -/
theorem parallel_vector_sum_diff (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_diff_l1180_118002


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l1180_118078

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 0.0009

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 0.018

/-- The weight of solution A in grams -/
def weight_A : ℝ := 200

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.0142

theorem liquid_X_percentage :
  percentage_X_in_A * weight_A + percentage_X_in_B * weight_B =
  percentage_X_in_mixture * (weight_A + weight_B) := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l1180_118078


namespace NUMINAMATH_CALUDE_items_left_in_store_l1180_118053

theorem items_left_in_store (ordered : ℕ) (sold : ℕ) (in_storeroom : ℕ) 
  (h_ordered : ordered = 4458)
  (h_sold : sold = 1561)
  (h_storeroom : in_storeroom = 575)
  (h_damaged : ⌊(5 : ℝ) / 100 * ordered⌋ = 222) : 
  ordered - sold - ⌊(5 : ℝ) / 100 * ordered⌋ + in_storeroom = 3250 := by
  sorry

end NUMINAMATH_CALUDE_items_left_in_store_l1180_118053


namespace NUMINAMATH_CALUDE_oldest_child_age_l1180_118019

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) (h4 : c ≥ b) (h5 : b ≥ a) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1180_118019


namespace NUMINAMATH_CALUDE_shopping_tax_free_cost_l1180_118054

/-- Given a shopping trip with a total spend, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
theorem shopping_tax_free_cost
  (total_spend : ℚ)
  (sales_tax : ℚ)
  (tax_rate : ℚ)
  (h1 : total_spend = 40)
  (h2 : sales_tax = 3/10)
  (h3 : tax_rate = 6/100)
  : ∃ (tax_free_cost : ℚ), tax_free_cost = 35 :=
by
  sorry


end NUMINAMATH_CALUDE_shopping_tax_free_cost_l1180_118054


namespace NUMINAMATH_CALUDE_choir_average_age_l1180_118069

theorem choir_average_age
  (num_females : ℕ)
  (num_males : ℕ)
  (avg_age_females : ℚ)
  (avg_age_males : ℚ)
  (total_people : ℕ)
  (h1 : num_females = 12)
  (h2 : num_males = 15)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 35)
  (h5 : total_people = num_females + num_males) :
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 31.89 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l1180_118069


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l1180_118080

theorem product_of_successive_numbers : 
  let x : ℝ := 97.49871794028884
  let y : ℝ := x + 1
  abs (x * y - 9603) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l1180_118080


namespace NUMINAMATH_CALUDE_always_integer_l1180_118087

theorem always_integer (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) :
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) / (k + 2 : ℤ) * (n.choose k) = m :=
sorry

end NUMINAMATH_CALUDE_always_integer_l1180_118087


namespace NUMINAMATH_CALUDE_car_speed_problem_l1180_118032

/-- Proves that given the conditions of the car problem, the speed of Car B is 50 km/h -/
theorem car_speed_problem (speed_b : ℝ) : 
  let speed_a := 3 * speed_b
  let time_a := 6
  let time_b := 2
  let total_distance := 1000
  speed_a * time_a + speed_b * time_b = total_distance →
  speed_b = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1180_118032


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1180_118073

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l1180_118073


namespace NUMINAMATH_CALUDE_cord_cutting_problem_l1180_118097

theorem cord_cutting_problem (cord1 : ℕ) (cord2 : ℕ) 
  (h1 : cord1 = 15) (h2 : cord2 = 12) : 
  Nat.gcd cord1 cord2 = 3 := by
sorry

end NUMINAMATH_CALUDE_cord_cutting_problem_l1180_118097


namespace NUMINAMATH_CALUDE_rice_quantity_calculation_rice_quantity_proof_l1180_118046

/-- Calculates the final quantity of rice that can be bought given initial conditions and price changes -/
theorem rice_quantity_calculation (initial_quantity : ℝ) 
  (first_price_reduction : ℝ) (second_price_reduction : ℝ) 
  (kg_to_pound_ratio : ℝ) (currency_exchange_rate : ℝ) : ℝ :=
  let after_first_reduction := initial_quantity * (1 / (1 - first_price_reduction))
  let after_second_reduction := after_first_reduction * (1 / (1 - second_price_reduction))
  let in_pounds := after_second_reduction * kg_to_pound_ratio
  let after_exchange_rate := in_pounds * (1 + currency_exchange_rate)
  let final_quantity := after_exchange_rate / kg_to_pound_ratio
  final_quantity

/-- The final quantity of rice that can be bought is approximately 29.17 kg -/
theorem rice_quantity_proof :
  ∃ ε > 0, |rice_quantity_calculation 20 0.2 0.1 2.2 0.05 - 29.17| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_rice_quantity_calculation_rice_quantity_proof_l1180_118046


namespace NUMINAMATH_CALUDE_trebled_resultant_l1180_118052

theorem trebled_resultant (initial_number : ℕ) : initial_number = 5 → 
  3 * (2 * initial_number + 15) = 75 := by
  sorry

end NUMINAMATH_CALUDE_trebled_resultant_l1180_118052


namespace NUMINAMATH_CALUDE_angle_through_point_l1180_118014

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) → 
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l1180_118014


namespace NUMINAMATH_CALUDE_possible_values_of_x_l1180_118063

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem possible_values_of_x (x : ℝ) :
  A x ∩ B x = B x → x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l1180_118063


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1180_118025

theorem sum_of_fractions : (1 : ℚ) / 3 + 2 / 7 + 3 / 8 = 167 / 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1180_118025


namespace NUMINAMATH_CALUDE_max_a_is_three_l1180_118076

/-- The function f(x) = x^3 - ax is monotonically increasing on [1, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → (x^3 - a*x) ≤ (y^3 - a*y)

/-- The maximum value of 'a' for which f(x) = x^3 - ax is monotonically increasing on [1, +∞) is 3 -/
theorem max_a_is_three :
  (∃ a_max : ℝ, a_max = 3 ∧
    (∀ a : ℝ, is_monotone_increasing a → a ≤ a_max) ∧
    is_monotone_increasing a_max) :=
sorry

end NUMINAMATH_CALUDE_max_a_is_three_l1180_118076


namespace NUMINAMATH_CALUDE_self_employed_tax_calculation_l1180_118086

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- The problem statement --/
theorem self_employed_tax_calculation :
  let income : ℝ := 350000
  let tax_rate : ℝ := 0.06
  calculate_tax_amount income tax_rate = 21000 := by
  sorry

end NUMINAMATH_CALUDE_self_employed_tax_calculation_l1180_118086


namespace NUMINAMATH_CALUDE_three_greater_than_negative_five_l1180_118081

theorem three_greater_than_negative_five :
  3 > -5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_five_l1180_118081


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l1180_118067

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (flagstaff_height_pos : 0 < flagstaff_height)
  (flagstaff_shadow_pos : 0 < flagstaff_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h_flagstaff : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  flagstaff_height / flagstaff_shadow * building_shadow = 12.4375 := by
sorry

end NUMINAMATH_CALUDE_building_height_from_shadows_l1180_118067


namespace NUMINAMATH_CALUDE_rainfall_solution_l1180_118088

def rainfall_problem (day1 day2 day3 : ℝ) : Prop :=
  day1 = 4 ∧
  day2 = 5 * day1 ∧
  day3 = day1 + day2 - 6

theorem rainfall_solution :
  ∀ day1 day2 day3 : ℝ,
  rainfall_problem day1 day2 day3 →
  day3 = 18 := by
sorry

end NUMINAMATH_CALUDE_rainfall_solution_l1180_118088


namespace NUMINAMATH_CALUDE_sqrt_2700_minus_37_cube_l1180_118005

theorem sqrt_2700_minus_37_cube (a b : ℕ+) :
  (Real.sqrt 2700 - 37 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2700_minus_37_cube_l1180_118005


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l1180_118033

theorem triangle_area_from_squares (a b c : ℝ) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l1180_118033


namespace NUMINAMATH_CALUDE_patrol_officer_results_l1180_118079

/-- Represents the travel record of the patrol officer -/
def travel_record : List Int := [10, -8, 6, -13, 7, -12, 3, -3]

/-- Position of the gas station relative to the guard post -/
def gas_station_position : Int := 6

/-- Fuel consumption rate of the motorcycle in liters per kilometer -/
def fuel_consumption_rate : ℚ := 0.05

/-- Calculates the final position of the patrol officer relative to the guard post -/
def final_position (record : List Int) : Int :=
  record.sum

/-- Counts the number of times the patrol officer passes the gas station -/
def gas_station_passes (record : List Int) (gas_station_pos : Int) : Nat :=
  sorry

/-- Calculates the total distance traveled by the patrol officer -/
def total_distance (record : List Int) : Int :=
  record.map (Int.natAbs) |>.sum

/-- Calculates the total fuel consumed during the patrol -/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  rate * distance.toNat

theorem patrol_officer_results :
  (final_position travel_record = -10) ∧
  (gas_station_passes travel_record gas_station_position = 4) ∧
  (total_fuel_consumed (total_distance travel_record) fuel_consumption_rate = 3.1) :=
sorry

end NUMINAMATH_CALUDE_patrol_officer_results_l1180_118079


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1180_118075

theorem quadratic_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → x^2 - 2*a*x + a + 2 ≥ 0) →
  a ∈ Set.Icc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l1180_118075


namespace NUMINAMATH_CALUDE_complex_number_equality_l1180_118038

theorem complex_number_equality : (1 + Complex.I)^10 / (1 - Complex.I) = -16 + 16 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1180_118038


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1180_118018

theorem consecutive_integers_product (a : ℤ) (h : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7 = 20) :
  (a + 6) * a = 391 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1180_118018


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1180_118061

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1180_118061


namespace NUMINAMATH_CALUDE_josh_film_purchase_l1180_118091

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def cost_per_film : ℕ := 5

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 4

/-- The cost of each CD in dollars -/
def cost_per_cd : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

/-- Theorem stating that the number of films Josh bought is correct -/
theorem josh_film_purchase :
  num_films * cost_per_film + num_books * cost_per_book + num_cds * cost_per_cd = total_spent :=
by sorry

end NUMINAMATH_CALUDE_josh_film_purchase_l1180_118091


namespace NUMINAMATH_CALUDE_vaccine_waiting_time_l1180_118072

/-- 
Given the waiting times for vaccine appointments and the total waiting time,
prove that the time waited after the second appointment is 14 days.
-/
theorem vaccine_waiting_time 
  (first_appointment_wait : ℕ) 
  (second_appointment_wait : ℕ)
  (total_wait : ℕ)
  (h1 : first_appointment_wait = 4)
  (h2 : second_appointment_wait = 20)
  (h3 : total_wait = 38) :
  total_wait - (first_appointment_wait + second_appointment_wait) = 14 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_waiting_time_l1180_118072


namespace NUMINAMATH_CALUDE_no_solution_condition_l1180_118048

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (m * x) / (x - 3) ≠ 3 / (x - 3)) ↔ (m = 1 ∨ m = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1180_118048


namespace NUMINAMATH_CALUDE_sum_of_matching_indices_l1180_118016

def sequence_length : ℕ := 1011

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_odds (n : ℕ) : ℕ := ((n + 1) / 2) ^ 2

theorem sum_of_matching_indices :
  sum_of_odds sequence_length = 256036 :=
sorry

end NUMINAMATH_CALUDE_sum_of_matching_indices_l1180_118016


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1180_118066

theorem geometric_sequence_sum_ratio 
  (a q : ℝ) 
  (h_q : q ≠ 1) : 
  let S : ℕ → ℝ := λ n => a * (1 - q^n) / (1 - q)
  (S 6 / S 3 = 1 / 2) → (S 9 / S 3 = 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1180_118066


namespace NUMINAMATH_CALUDE_smallest_integers_difference_l1180_118057

theorem smallest_integers_difference : ∃ (n₁ n₂ : ℕ), 
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 4620 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_l1180_118057
