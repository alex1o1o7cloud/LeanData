import Mathlib

namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l3281_328131

/-- The minimum translation that makes sin(3x + π/6) symmetric about y-axis -/
theorem min_translation_for_symmetry :
  let f (x : ℝ) := Real.sin (3 * x + π / 6)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x : ℝ), f (x - m) = f (-x - m) ∨ f (x + m) = f (-x + m)) ∧
    (∀ (m' : ℝ), m' > 0 → 
      (∀ (x : ℝ), f (x - m') = f (-x - m') ∨ f (x + m') = f (-x + m')) →
      m ≤ m') ∧
    m = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l3281_328131


namespace NUMINAMATH_CALUDE_custom_op_negative_four_six_l3281_328177

/-- Custom binary operation "*" -/
def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

/-- Theorem stating that (-4) * 6 = 68 under the custom operation -/
theorem custom_op_negative_four_six : custom_op (-4) 6 = 68 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_negative_four_six_l3281_328177


namespace NUMINAMATH_CALUDE_system_solution_l3281_328179

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = -20) ∧ 
  (x = -34/13) ∧ (y = -15/13) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3281_328179


namespace NUMINAMATH_CALUDE_cuboid_max_volume_l3281_328159

theorem cuboid_max_volume (d : ℝ) (p : ℝ) (h1 : d = 10) (h2 : p = 8) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * b * c ≤ 192 ∧
  a^2 + b^2 + c^2 = d^2 ∧
  a^2 + b^2 = p^2 ∧
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + y^2 + z^2 = d^2 → x^2 + y^2 = p^2 → x * y * z ≤ 192) :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_max_volume_l3281_328159


namespace NUMINAMATH_CALUDE_seating_arrangement_l3281_328191

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def restricted_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

theorem seating_arrangement (n : ℕ) (k : ℕ) 
  (h1 : n = 8) (h2 : k = 4) : 
  total_arrangements n - restricted_arrangements n k = 37440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3281_328191


namespace NUMINAMATH_CALUDE_min_tiles_for_square_l3281_328103

def tile_length : ℕ := 6
def tile_width : ℕ := 4

def tile_area : ℕ := tile_length * tile_width

def square_side : ℕ := Nat.lcm tile_length tile_width

theorem min_tiles_for_square :
  (square_side * square_side) / tile_area = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_square_l3281_328103


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3281_328134

theorem sin_cos_identity (α : ℝ) : (Real.sin α - Real.cos α)^2 + Real.sin (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3281_328134


namespace NUMINAMATH_CALUDE_donghwan_candies_l3281_328157

theorem donghwan_candies (total_candies bag_size : ℕ) 
  (h1 : total_candies = 138)
  (h2 : bag_size = 18) :
  total_candies % bag_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_donghwan_candies_l3281_328157


namespace NUMINAMATH_CALUDE_zhang_qiujian_problem_l3281_328140

theorem zhang_qiujian_problem (x y : ℤ) : 
  (x + 10 - (y - 10) = 5 * (y - 10) ∧ x - 10 = y + 10) ↔
  (x = y + 10 ∧ 
   x + 10 - (y - 10) = 5 * (y - 10) ∧ 
   x - 10 = y + 10) :=
by sorry

end NUMINAMATH_CALUDE_zhang_qiujian_problem_l3281_328140


namespace NUMINAMATH_CALUDE_fifth_house_gnomes_l3281_328185

/-- The number of houses on the street -/
def num_houses : Nat := 5

/-- The number of gnomes in each of the first four houses -/
def gnomes_per_house : Nat := 3

/-- The total number of gnomes on the street -/
def total_gnomes : Nat := 20

/-- The number of gnomes in the fifth house -/
def gnomes_in_fifth_house : Nat := total_gnomes - (4 * gnomes_per_house)

theorem fifth_house_gnomes :
  gnomes_in_fifth_house = 8 := by sorry

end NUMINAMATH_CALUDE_fifth_house_gnomes_l3281_328185


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3281_328173

theorem arithmetic_calculation : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3281_328173


namespace NUMINAMATH_CALUDE_max_n_for_int_polynomial_l3281_328164

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that P(aᵢ) = i for all 1 ≤ i ≤ n -/
def SatisfiesProperty (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℤ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → P.eval (a i) = i

/-- The theorem stating the maximum n for which the property holds -/
theorem max_n_for_int_polynomial (P : IntPolynomial) (h : P.degree = 2022) :
    (∃ n : ℕ, SatisfiesProperty P n ∧ ∀ m : ℕ, SatisfiesProperty P m → m ≤ n) ∧
    (∃ n : ℕ, n = 2022 ∧ SatisfiesProperty P n) :=
  sorry

end NUMINAMATH_CALUDE_max_n_for_int_polynomial_l3281_328164


namespace NUMINAMATH_CALUDE_fruit_orders_eq_six_l3281_328110

/-- Represents the types of fruit in the basket -/
inductive Fruit
  | Apple
  | Peach
  | Pear

/-- The number of fruits in the basket -/
def basket_size : Nat := 3

/-- The number of chances to draw -/
def draw_chances : Nat := 2

/-- Calculates the number of different orders of fruit that can be drawn -/
def fruit_orders : Nat :=
  basket_size * (basket_size - 1)

theorem fruit_orders_eq_six :
  fruit_orders = 6 :=
sorry

end NUMINAMATH_CALUDE_fruit_orders_eq_six_l3281_328110


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3281_328119

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 8) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧
    P = 7/15 ∧ Q = -4/3 ∧ R = 14/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3281_328119


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_equivalence_l3281_328124

theorem knicks_knacks_knocks_equivalence 
  (h1 : (8 : ℚ) * knicks = (3 : ℚ) * knacks)
  (h2 : (4 : ℚ) * knacks = (5 : ℚ) * knocks) :
  (64 : ℚ) * knicks = (30 : ℚ) * knocks := by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_equivalence_l3281_328124


namespace NUMINAMATH_CALUDE_prime_power_square_sum_l3281_328170

theorem prime_power_square_sum (p n k : ℕ) : 
  p.Prime → p > 0 → n > 0 → k > 0 → 144 + p^n = k^2 →
  ((p = 5 ∧ n = 2 ∧ k = 13) ∨ (p = 2 ∧ n = 8 ∧ k = 20) ∨ (p = 3 ∧ n = 4 ∧ k = 15)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_square_sum_l3281_328170


namespace NUMINAMATH_CALUDE_matrix_inverse_problem_l3281_328148

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_inverse_problem (B : Matrix n n ℝ) (h_inv : Invertible B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) :
  B + 10 • B⁻¹ = (160 / 15 : ℝ) • 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_problem_l3281_328148


namespace NUMINAMATH_CALUDE_cost_to_fly_AB_l3281_328100

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The distance from A to B in kilometers -/
def distance_AB : ℝ := 4500

theorem cost_to_fly_AB : flying_cost distance_AB = 660 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_fly_AB_l3281_328100


namespace NUMINAMATH_CALUDE_distance_between_points_l3281_328107

/-- The distance between points (0,15,5) and (8,0,12) in 3D space is √338. -/
theorem distance_between_points : Real.sqrt 338 = Real.sqrt ((8 - 0)^2 + (0 - 15)^2 + (12 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3281_328107


namespace NUMINAMATH_CALUDE_nori_initial_boxes_l3281_328114

/-- The number of crayons in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayons Nori gave to Mae -/
def crayons_to_mae : ℕ := 5

/-- The additional number of crayons Nori gave to Lea compared to Mae -/
def additional_crayons_to_lea : ℕ := 7

/-- The number of crayons Nori has left -/
def crayons_left : ℕ := 15

/-- The number of boxes Nori had initially -/
def initial_boxes : ℕ := 4

theorem nori_initial_boxes : 
  crayons_per_box * initial_boxes = 
    crayons_left + crayons_to_mae + (crayons_to_mae + additional_crayons_to_lea) :=
by sorry

end NUMINAMATH_CALUDE_nori_initial_boxes_l3281_328114


namespace NUMINAMATH_CALUDE_m_value_in_set_union_l3281_328109

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem m_value_in_set_union (m : ℝ) :
  A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_value_in_set_union_l3281_328109


namespace NUMINAMATH_CALUDE_die_probability_l3281_328106

/-- The number of times the die is tossed -/
def n : ℕ := 30

/-- The number of faces on the die -/
def faces : ℕ := 6

/-- The number of favorable outcomes before the first six -/
def favorable_before : ℕ := 3

/-- Probability of the event: at least one six appears, and no five or four appears before the first six -/
def prob_event : ℚ :=
  1 / 3

theorem die_probability :
  prob_event = (favorable_before ^ (n - 1) * (2 ^ n - 1)) / (faces ^ n) :=
sorry

end NUMINAMATH_CALUDE_die_probability_l3281_328106


namespace NUMINAMATH_CALUDE_largest_term_binomial_sequence_l3281_328112

theorem largest_term_binomial_sequence (k : ℕ) :
  k ≤ 1992 →
  k * Nat.choose 1992 k ≤ 997 * Nat.choose 1992 997 :=
by sorry

end NUMINAMATH_CALUDE_largest_term_binomial_sequence_l3281_328112


namespace NUMINAMATH_CALUDE_garbage_collection_l3281_328181

theorem garbage_collection (D : ℝ) : 
  (∃ (Dewei Zane : ℝ), 
    Dewei = D - 2 ∧ 
    Zane = 4 * Dewei ∧ 
    Zane = 62) → 
  D = 17.5 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_l3281_328181


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l3281_328160

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Carol's schedule cycle length -/
def carol_cycle : ℕ := 6

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest days in a cycle -/
def al_rest_days : Finset ℕ := {5, 6}

/-- Carol's rest days in a cycle -/
def carol_rest_days : Finset ℕ := {6}

/-- The number of days both Al and Carol have rest-days on the same day -/
def coinciding_rest_days : ℕ := (al_rest_days ∩ carol_rest_days).card * (total_days / al_cycle)

theorem coinciding_rest_days_count : coinciding_rest_days = 166 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l3281_328160


namespace NUMINAMATH_CALUDE_invisible_square_existence_l3281_328144

theorem invisible_square_existence (n : ℕ+) :
  ∃ (x y : ℤ), ∀ (i j : ℕ), 0 < i ∧ i ≤ n ∧ 0 < j ∧ j ≤ n →
    Nat.gcd (Int.natAbs (x + i)) (Int.natAbs (y + j)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_existence_l3281_328144


namespace NUMINAMATH_CALUDE_quadratic_function_k_l3281_328120

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_k (a b c : ℤ) : 
  (f a b c 1 = 0) →
  (60 < f a b c 6 ∧ f a b c 6 < 70) →
  (120 < f a b c 9 ∧ f a b c 9 < 130) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1) ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_l3281_328120


namespace NUMINAMATH_CALUDE_jesses_room_length_l3281_328117

theorem jesses_room_length (area : ℝ) (width : ℝ) (h1 : area = 12.0) (h2 : width = 8) :
  area / width = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_length_l3281_328117


namespace NUMINAMATH_CALUDE_state_university_cost_l3281_328171

theorem state_university_cost (tuition room_and_board total_cost : ℕ) : 
  tuition = 1644 →
  tuition = room_and_board + 704 →
  total_cost = tuition + room_and_board →
  total_cost = 2584 := by
  sorry

end NUMINAMATH_CALUDE_state_university_cost_l3281_328171


namespace NUMINAMATH_CALUDE_roger_trays_capacity_l3281_328190

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := 12

theorem roger_trays_capacity :
  trays_per_trip = 4 ∧ 
  num_trips * trays_per_trip = total_trays :=
by sorry

end NUMINAMATH_CALUDE_roger_trays_capacity_l3281_328190


namespace NUMINAMATH_CALUDE_matching_jelly_bean_probability_l3281_328195

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 1, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 2, red := 3, blue := 0, yellow := 2 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matching_color_probability (person1 person2 : JellyBeans) : ℚ :=
  let total1 := total_jelly_beans person1
  let total2 := total_jelly_beans person2
  (person1.green * person2.green + person1.red * person2.red + person1.blue * person2.blue) / (total1 * total2)

theorem matching_jelly_bean_probability :
  matching_color_probability abe_jelly_beans bob_jelly_beans = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_matching_jelly_bean_probability_l3281_328195


namespace NUMINAMATH_CALUDE_merill_marbles_vivian_marbles_l3281_328178

-- Define the number of marbles for each person
def Selma : ℕ := 50
def Elliot : ℕ := 15
def Merill : ℕ := 2 * Elliot
def Vivian : ℕ := 21

-- Theorem to prove Merill's marbles
theorem merill_marbles : Merill = 30 := by sorry

-- Theorem to prove Vivian's marbles
theorem vivian_marbles : Vivian = 21 ∧ Vivian > Elliot + 5 ∧ Vivian ≥ (135 * Elliot) / 100 := by sorry

end NUMINAMATH_CALUDE_merill_marbles_vivian_marbles_l3281_328178


namespace NUMINAMATH_CALUDE_function_property_l3281_328167

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_sym : ∀ x, f (3/2 + x) = -f (3/2 - x))
    (h_f1 : f 1 = 2) : 
  f 2 + f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3281_328167


namespace NUMINAMATH_CALUDE_a_2018_value_l3281_328155

def triangle_sequence (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem a_2018_value (A : ℕ → ℝ) 
  (h1 : ∀ n, triangle_sequence (triangle_sequence A) n = 1)
  (h2 : A 18 = 0)
  (h3 : A 2017 = 0) :
  A 2018 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_2018_value_l3281_328155


namespace NUMINAMATH_CALUDE_alfonso_daily_earnings_l3281_328135

def helmet_cost : ℕ := 340
def savings : ℕ := 40
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10

def total_working_days : ℕ := days_per_week * total_weeks

def additional_savings_needed : ℕ := helmet_cost - savings

theorem alfonso_daily_earnings :
  additional_savings_needed / total_working_days = 6 :=
by sorry

end NUMINAMATH_CALUDE_alfonso_daily_earnings_l3281_328135


namespace NUMINAMATH_CALUDE_steve_initial_boxes_l3281_328152

/-- The number of boxes Steve had initially -/
def initial_boxes (pencils_per_box : ℕ) (pencils_to_lauren : ℕ) (pencils_to_matt_diff : ℕ) (pencils_left : ℕ) : ℕ :=
  (pencils_to_lauren + (pencils_to_lauren + pencils_to_matt_diff) + pencils_left) / pencils_per_box

theorem steve_initial_boxes :
  initial_boxes 12 6 3 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_steve_initial_boxes_l3281_328152


namespace NUMINAMATH_CALUDE_sum_plus_five_mod_seven_l3281_328145

/-- Sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The problem statement -/
theorem sum_plus_five_mod_seven :
  (sum_to_n 99 + 5) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_five_mod_seven_l3281_328145


namespace NUMINAMATH_CALUDE_daniels_horses_l3281_328176

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

theorem daniels_horses :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = total_legs :=
by sorry

end NUMINAMATH_CALUDE_daniels_horses_l3281_328176


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3281_328197

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 30 years older than his son and the son's current age is 28 years. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 28 →
    man_age = son_age + 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3281_328197


namespace NUMINAMATH_CALUDE_tims_bodyguard_cost_l3281_328132

/-- Calculate the total weekly cost for bodyguards --/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Prove that the total weekly cost for Tim's bodyguards is $2240 --/
theorem tims_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_tims_bodyguard_cost_l3281_328132


namespace NUMINAMATH_CALUDE_faculty_marriage_percentage_l3281_328188

theorem faculty_marriage_percentage (total : ℕ) (total_pos : 0 < total) : 
  let women := (70 : ℚ) / 100 * total
  let men := total - women
  let single_men := (1 : ℚ) / 3 * men
  let married_men := (2 : ℚ) / 3 * men
  (married_men : ℚ) / total ≥ (20 : ℚ) / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_faculty_marriage_percentage_l3281_328188


namespace NUMINAMATH_CALUDE_parabola_vertex_l3281_328174

/-- The parabola is defined by the equation y = 2(x-5)^2 + 3 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 5)^2 + 3

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x₀ y₀ : ℝ) : Prop :=
  ∀ x y, parabola x y → y ≥ y₀

/-- Theorem: The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex :
  is_vertex 5 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3281_328174


namespace NUMINAMATH_CALUDE_days_to_watch_all_episodes_l3281_328165

-- Define the number of episodes for each season type
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_episodes : ℕ := regular_season_episodes + 4

-- Define the duration of episodes for different seasons
def early_episode_duration : ℚ := 1/2
def later_episode_duration : ℚ := 3/4

-- Define John's daily watching time
def daily_watching_time : ℚ := 2

-- Define the total number of seasons
def total_seasons : ℕ := 10

-- Define a function to calculate the total viewing time
def total_viewing_time : ℚ :=
  let early_seasons_episodes : ℕ := 2 * regular_season_episodes + third_season_episodes
  let early_seasons_time : ℚ := early_seasons_episodes * early_episode_duration
  let later_seasons_episodes : ℕ := (total_seasons - 3) * regular_season_episodes + last_season_episodes
  let later_seasons_time : ℚ := later_seasons_episodes * later_episode_duration
  early_seasons_time + later_seasons_time

-- Theorem statement
theorem days_to_watch_all_episodes :
  ⌈total_viewing_time / daily_watching_time⌉ = 77 := by sorry

end NUMINAMATH_CALUDE_days_to_watch_all_episodes_l3281_328165


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3281_328136

-- Define the polynomial
def p (x : ℝ) : ℝ := 42 * x^3 - 35 * x^2 + 10 * x - 1

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocal :
  p a = 0 ∧ p b = 0 ∧ p c = 0 ∧   -- a, b, c are roots of p
  0 < a ∧ a < 1 ∧                 -- a is between 0 and 1
  0 < b ∧ b < 1 ∧                 -- b is between 0 and 1
  0 < c ∧ c < 1 ∧                 -- c is between 0 and 1
  a ≠ b ∧ b ≠ c ∧ a ≠ c →         -- roots are distinct
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2.875 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3281_328136


namespace NUMINAMATH_CALUDE_rajan_income_l3281_328123

/-- Represents the financial situation of two individuals -/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income based on the given financial situation -/
def calculate_income (fs : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that given the specific financial situation, Rajan's income is 7000 -/
theorem rajan_income (fs : FinancialSituation) 
  (h1 : fs.income_ratio = 7/6)
  (h2 : fs.expenditure_ratio = 6/5)
  (h3 : fs.savings = 1000) :
  calculate_income fs = 7000 := by
  sorry

end NUMINAMATH_CALUDE_rajan_income_l3281_328123


namespace NUMINAMATH_CALUDE_a_10_value_l3281_328141

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem a_10_value (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 7 = 10 → q = -2 → a 10 = -80 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l3281_328141


namespace NUMINAMATH_CALUDE_rd_scenario_theorem_l3281_328168

/-- Represents a firm in the R&D scenario -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario -/
structure RDScenario where
  V : ℝ  -- Revenue if successful
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- The main theorem to prove -/
theorem rd_scenario_theorem (s : RDScenario) 
    (h1 : 0 < s.α ∧ s.α < 1) 
    (h2 : s.V > 0) 
    (h3 : s.IC > 0) : 
  (bothParticipateCondition s ↔ expectedRevenueBoth s ≥ s.IC) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → bothParticipateCondition s) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → totalProfitOne s > totalProfitBoth s) := by
  sorry

end NUMINAMATH_CALUDE_rd_scenario_theorem_l3281_328168


namespace NUMINAMATH_CALUDE_cinema_hall_capacity_l3281_328180

/-- Represents a cinema hall with a given number of rows and seats per row -/
structure CinemaHall where
  rows : ℕ
  seatsPerRow : ℕ

/-- Calculates the approximate seating capacity of a cinema hall -/
def approximateCapacity (hall : CinemaHall) : ℕ :=
  900

/-- Calculates the actual seating capacity of a cinema hall -/
def actualCapacity (hall : CinemaHall) : ℕ :=
  hall.rows * hall.seatsPerRow

theorem cinema_hall_capacity (hall : CinemaHall) 
  (h1 : hall.rows = 28) 
  (h2 : hall.seatsPerRow = 31) : 
  approximateCapacity hall = 900 ∧ actualCapacity hall = 868 := by
  sorry

end NUMINAMATH_CALUDE_cinema_hall_capacity_l3281_328180


namespace NUMINAMATH_CALUDE_acute_angles_sum_l3281_328139

theorem acute_angles_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l3281_328139


namespace NUMINAMATH_CALUDE_inequality_proof_l3281_328182

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃) (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃) : 
  3 * (a₁ * b₁ + a₂ * b₂ + a₃ * b₃) ≥ (a₁ + a₂ + a₃) * (b₁ + b₂ + b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3281_328182


namespace NUMINAMATH_CALUDE_master_bedroom_size_l3281_328116

theorem master_bedroom_size (total_area guest_area master_area combined_area : ℝ) 
  (h1 : total_area = 2300)
  (h2 : combined_area = 1000)
  (h3 : guest_area = (1/4) * master_area)
  (h4 : total_area = combined_area + guest_area + master_area) :
  master_area = 1040 := by
sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l3281_328116


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l3281_328199

theorem min_k_for_inequality (k : ℝ) : 
  (∀ x : ℝ, x > 0 → k * x ≥ (Real.sin x) / (2 + Real.cos x)) ↔ k ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l3281_328199


namespace NUMINAMATH_CALUDE_infinite_set_sum_of_digits_squared_equal_l3281_328101

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exists an infinite set of natural numbers n, not ending in 0,
    such that the sum of digits of n^2 equals the sum of digits of n -/
theorem infinite_set_sum_of_digits_squared_equal :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ n ∈ S, n % 10 ≠ 0 ∧ sum_of_digits (n^2) = sum_of_digits n) :=
sorry

end NUMINAMATH_CALUDE_infinite_set_sum_of_digits_squared_equal_l3281_328101


namespace NUMINAMATH_CALUDE_days_per_month_is_30_l3281_328122

/-- Represents the number of trees a single logger can cut down in one day. -/
def trees_per_logger_per_day : ℕ := 6

/-- Represents the length of the forest in miles. -/
def forest_length : ℕ := 4

/-- Represents the width of the forest in miles. -/
def forest_width : ℕ := 6

/-- Represents the number of trees in each square mile of the forest. -/
def trees_per_square_mile : ℕ := 600

/-- Represents the number of loggers working on cutting down the trees. -/
def num_loggers : ℕ := 8

/-- Represents the number of months it takes to cut down all trees. -/
def num_months : ℕ := 10

/-- Theorem stating that the number of days in each month is 30. -/
theorem days_per_month_is_30 :
  ∃ (days_per_month : ℕ),
    days_per_month = 30 ∧
    (forest_length * forest_width * trees_per_square_mile =
     num_loggers * trees_per_logger_per_day * num_months * days_per_month) :=
by sorry

end NUMINAMATH_CALUDE_days_per_month_is_30_l3281_328122


namespace NUMINAMATH_CALUDE_power_eight_sum_ratio_l3281_328118

theorem power_eight_sum_ratio (x y k : ℝ) 
  (h : (x^2 + y^2)/(x^2 - y^2) + (x^2 - y^2)/(x^2 + y^2) = k) :
  (x^8 + y^8)/(x^8 - y^8) + (x^8 - y^8)/(x^8 + y^8) = (k^4 + 24*k^2 + 16)/(4*k^3 + 16*k) :=
by sorry

end NUMINAMATH_CALUDE_power_eight_sum_ratio_l3281_328118


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3281_328149

theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (2 * (a - d)) * ((a + d) + 7) = a^2 ∧ 
  (a - d) * a * (a + d) = 1000 →
  d = 8 ∨ d = -15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3281_328149


namespace NUMINAMATH_CALUDE_jasons_leg_tattoos_l3281_328111

theorem jasons_leg_tattoos (jason_arm_tattoos : ℕ) (adam_tattoos : ℕ) :
  jason_arm_tattoos = 2 →
  adam_tattoos = 23 →
  ∃ (jason_leg_tattoos : ℕ),
    adam_tattoos = 2 * (2 * jason_arm_tattoos + 2 * jason_leg_tattoos) + 3 ∧
    jason_leg_tattoos = 3 :=
by sorry

end NUMINAMATH_CALUDE_jasons_leg_tattoos_l3281_328111


namespace NUMINAMATH_CALUDE_solve_system_l3281_328172

theorem solve_system (x y z w : ℤ)
  (eq1 : x + y = 4)
  (eq2 : x - y = 36)
  (eq3 : x * z + y * w = 50)
  (eq4 : z - w = 5) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3281_328172


namespace NUMINAMATH_CALUDE_z_squared_minus_four_z_is_real_l3281_328192

/-- Given a real number a and a complex number z = 2 + ai, 
    prove that z^2 - 4z is a real number. -/
theorem z_squared_minus_four_z_is_real (a : ℝ) : 
  let z : ℂ := 2 + a * Complex.I
  (z^2 - 4*z).im = 0 := by sorry

end NUMINAMATH_CALUDE_z_squared_minus_four_z_is_real_l3281_328192


namespace NUMINAMATH_CALUDE_tray_pieces_count_l3281_328161

def tray_length : ℕ := 24
def tray_width : ℕ := 20
def piece_length : ℕ := 3
def piece_width : ℕ := 2

theorem tray_pieces_count : 
  (tray_length * tray_width) / (piece_length * piece_width) = 80 :=
sorry

end NUMINAMATH_CALUDE_tray_pieces_count_l3281_328161


namespace NUMINAMATH_CALUDE_fourth_root_of_fourth_power_l3281_328130

theorem fourth_root_of_fourth_power (a : ℝ) (h : a < 2) :
  (((a - 2) ^ 4) ^ (1/4 : ℝ)) = 2 - a := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_fourth_power_l3281_328130


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3281_328186

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4) 
  (eq2 : 1 / x - 1 / y = -5) : 
  x + y = -16 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3281_328186


namespace NUMINAMATH_CALUDE_race_head_start_l3281_328143

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (20 / 14) * Vb) :
  ∃ H : ℝ, H = (3 / 10) * L ∧ L / Va = (L - H) / Vb :=
sorry

end NUMINAMATH_CALUDE_race_head_start_l3281_328143


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3281_328129

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = -3 ∧ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3281_328129


namespace NUMINAMATH_CALUDE_one_and_one_third_l3281_328158

theorem one_and_one_third : ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 135 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_l3281_328158


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l3281_328153

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (must_be_together : ℕ) : ℕ :=
  sorry

/-- The theorem to prove -/
theorem student_distribution_theorem :
  distribute_students 5 3 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l3281_328153


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l3281_328193

/-- Represents a lock with a specified number of dials and digits per dial -/
structure Lock :=
  (numDials : ℕ)
  (digitsPerDial : ℕ)

/-- Calculates the number of different settings for a lock with all digits different -/
def countDifferentSettings (lock : Lock) : ℕ :=
  sorry

/-- The specific lock in the problem -/
def suitcaseLock : Lock :=
  { numDials := 3,
    digitsPerDial := 10 }

/-- Theorem stating that the number of different settings for the suitcase lock is 720 -/
theorem suitcase_lock_settings :
  countDifferentSettings suitcaseLock = 720 :=
sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l3281_328193


namespace NUMINAMATH_CALUDE_complex_argument_cube_l3281_328187

theorem complex_argument_cube (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 5)
  (h3 : Complex.abs (z₁ + z₂) = 7) :
  Complex.arg ((z₂ / z₁) ^ 3) = π := by sorry

end NUMINAMATH_CALUDE_complex_argument_cube_l3281_328187


namespace NUMINAMATH_CALUDE_total_students_l3281_328105

def line_up (students_between : ℕ) (right_of_hoseok : ℕ) (left_of_yoongi : ℕ) : ℕ :=
  2 + students_between + right_of_hoseok + left_of_yoongi

theorem total_students :
  line_up 5 9 6 = 22 :=
by sorry

end NUMINAMATH_CALUDE_total_students_l3281_328105


namespace NUMINAMATH_CALUDE_fat_per_serving_perrys_recipe_fat_per_serving_l3281_328146

/-- Calculates the grams of fat per serving in a sauce recipe. -/
theorem fat_per_serving (servings : ℕ) (total_mixture : ℝ) 
  (cream_ratio cheese_ratio butter_ratio : ℕ) 
  (cream_fat cheese_fat butter_fat : ℝ) : ℝ :=
  let total_ratio := cream_ratio + cheese_ratio + butter_ratio
  let part_size := total_mixture / total_ratio
  let cream_amount := part_size * cream_ratio
  let cheese_amount := part_size * cheese_ratio
  let butter_amount := part_size * butter_ratio * 2 -- Convert half-cups to cups
  let total_fat := cream_amount * cream_fat + cheese_amount * cheese_fat + butter_amount * butter_fat
  total_fat / servings

/-- The amount of fat per serving in Perry's recipe is approximately 37.65 grams. -/
theorem perrys_recipe_fat_per_serving : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |fat_per_serving 6 1.5 5 3 2 88 110 184 - 37.65| < ε :=
sorry

end NUMINAMATH_CALUDE_fat_per_serving_perrys_recipe_fat_per_serving_l3281_328146


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l3281_328142

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ)
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : extra_bananas = 2) :
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l3281_328142


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_abc_l3281_328196

def is_valid_abc (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = 2) ∧ (n % 10 = 7)

def largest_abc : ℕ := 297
def smallest_abc : ℕ := 207

theorem sum_of_largest_and_smallest_abc :
  is_valid_abc largest_abc ∧
  is_valid_abc smallest_abc ∧
  (∀ n : ℕ, is_valid_abc n → smallest_abc ≤ n ∧ n ≤ largest_abc) ∧
  largest_abc + smallest_abc = 504 :=
sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_abc_l3281_328196


namespace NUMINAMATH_CALUDE_complement_union_original_equals_universe_l3281_328128

-- Define the universe set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set C as the complement of M in U
def C : Finset Nat := U \ M

-- Theorem statement
theorem complement_union_original_equals_universe :
  C ∪ M = U := by sorry

end NUMINAMATH_CALUDE_complement_union_original_equals_universe_l3281_328128


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3281_328127

/-- Given a hyperbola and a line intersecting it, proves that the eccentricity is √2 under specific conditions -/
theorem hyperbola_eccentricity (a b k m : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (A B N : ℝ × ℝ), 
    -- The line y = kx + m intersects the hyperbola at A and B
    (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.2 = k * A.1 + m) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.2 = k * B.1 + m) ∧
    -- A and B are where the asymptotes intersect the line
    (A.2 = -b/a * A.1 ∨ A.2 = b/a * A.1) ∧
    (B.2 = -b/a * B.1 ∨ B.2 = b/a * B.1) ∧
    -- N is on both lines
    (N.2 = k * N.1 + m) ∧
    (N.2 = 1/k * N.1) ∧
    -- N is the midpoint of AB
    (N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2)) →
  -- The eccentricity of the hyperbola is √2
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3281_328127


namespace NUMINAMATH_CALUDE_function_inequality_l3281_328183

theorem function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x, (x - 1) * (deriv f x) < 0) : 
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3281_328183


namespace NUMINAMATH_CALUDE_expression_evaluation_l3281_328194

theorem expression_evaluation : 2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3281_328194


namespace NUMINAMATH_CALUDE_cookies_packs_l3281_328126

theorem cookies_packs (total packs_cake packs_chocolate : ℕ) 
  (h1 : total = 42)
  (h2 : packs_cake = 22)
  (h3 : packs_chocolate = 16) :
  total - packs_cake - packs_chocolate = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookies_packs_l3281_328126


namespace NUMINAMATH_CALUDE_chord_length_on_xaxis_l3281_328151

/-- The length of the chord intercepted by the x-axis on the circle (x-1)^2+(y-1)^2=2 is 2 -/
theorem chord_length_on_xaxis (x y : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    ((x₁ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    ((x₂ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    (x₂ - x₁ = 2)) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_on_xaxis_l3281_328151


namespace NUMINAMATH_CALUDE_lcm_triple_count_l3281_328184

/-- The least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a, b, c) satisfying the LCM conditions -/
def count_triples : ℕ := sorry

/-- Main theorem: There are exactly 70 ordered triples satisfying the LCM conditions -/
theorem lcm_triple_count :
  count_triples = 70 :=
by sorry

end NUMINAMATH_CALUDE_lcm_triple_count_l3281_328184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3281_328138

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 - sum_arithmetic_sequence a₁ d 10 / 10 = 2) →
  (sum_arithmetic_sequence (-2008) d 2008 = -2008) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3281_328138


namespace NUMINAMATH_CALUDE_correct_average_l3281_328108

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 19 ∧ incorrect_num = 26 ∧ correct_num = 76 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3281_328108


namespace NUMINAMATH_CALUDE_tan_pi_fourth_plus_alpha_l3281_328162

theorem tan_pi_fourth_plus_alpha (α : Real) (h : Real.tan α = 2) : 
  Real.tan (π/4 + α) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_fourth_plus_alpha_l3281_328162


namespace NUMINAMATH_CALUDE_mn_square_value_l3281_328189

theorem mn_square_value (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  (m + n)^2 = 1 ∨ (m + n)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_mn_square_value_l3281_328189


namespace NUMINAMATH_CALUDE_number_difference_l3281_328175

theorem number_difference (S L : ℕ) (h1 : S = 270) (h2 : L = 6 * S + 15) :
  L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3281_328175


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3281_328154

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a + 2) * (a - 3) = 20 ∧ (b + 2) * (b - 3) = 20 ∧ a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3281_328154


namespace NUMINAMATH_CALUDE_number_of_schnauzers_l3281_328104

theorem number_of_schnauzers : ℕ := by
  -- Define the number of Doberman puppies
  let doberman : ℕ := 20

  -- Define the equation from the problem
  let equation (s : ℕ) : Prop := 3 * doberman - 5 + (doberman - s) = 90

  -- Assert that the equation holds for s = 55
  have h : equation 55 := by sorry

  -- Prove that 55 is the unique solution
  have unique : ∀ s : ℕ, equation s → s = 55 := by sorry

  -- Conclude that the number of Schnauzers is 55
  exact 55

end NUMINAMATH_CALUDE_number_of_schnauzers_l3281_328104


namespace NUMINAMATH_CALUDE_smallest_divisor_after_429_l3281_328137

theorem smallest_divisor_after_429 (n : ℕ) : 
  10000 ≤ n ∧ n < 100000 →  -- n is a five-digit number
  429 ∣ n →                 -- 429 is a divisor of n
  ∃ d : ℕ, d ∣ n ∧ 429 < d ∧ d ≤ 858 ∧ 
    ∀ d' : ℕ, d' ∣ n → 429 < d' → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_429_l3281_328137


namespace NUMINAMATH_CALUDE_polynomial_no_integral_roots_l3281_328102

/-- A polynomial with integral coefficients that has odd integer values at 0 and 1 has no integral roots. -/
theorem polynomial_no_integral_roots 
  (p : Polynomial ℤ) 
  (h0 : Odd (p.eval 0)) 
  (h1 : Odd (p.eval 1)) : 
  ∀ (x : ℤ), p.eval x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_integral_roots_l3281_328102


namespace NUMINAMATH_CALUDE_four_tangent_circles_l3281_328125

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum of their radii --/
def are_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- A circle is tangent to two other circles --/
def is_tangent_to_both (c c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

theorem four_tangent_circles (c1 c2 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), s.card = 4 ∧ ∀ c ∈ s, c.radius = 3 ∧ is_tangent_to_both c c1 c2 :=
sorry

end NUMINAMATH_CALUDE_four_tangent_circles_l3281_328125


namespace NUMINAMATH_CALUDE_external_polygon_sides_l3281_328150

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons as described in the problem -/
structure PolygonArrangement :=
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the resulting external polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.hexagon.sides +
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.heptagon.sides +
  arrangement.nonagon.sides -
  10 -- Subtracting the sides that are shared between polygons

/-- The main theorem stating that the resulting external polygon has 20 sides -/
theorem external_polygon_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.hexagon.sides = 6)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.square.sides = 4)
  (h4 : arrangement.pentagon.sides = 5)
  (h5 : arrangement.heptagon.sides = 7)
  (h6 : arrangement.nonagon.sides = 9) :
  exposedSides arrangement = 20 := by
  sorry

end NUMINAMATH_CALUDE_external_polygon_sides_l3281_328150


namespace NUMINAMATH_CALUDE_value_of_a_l3281_328133

theorem value_of_a (x y a : ℝ) 
  (h1 : 3^x = a) 
  (h2 : 5^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3281_328133


namespace NUMINAMATH_CALUDE_forgotten_angles_sum_l3281_328121

/-- The sum of interior angles of a polygon with n sides --/
def polygon_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- A convex polygon with n sides where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

theorem forgotten_angles_sum (p : ConvexPolygon) 
  (partial_sum : ℝ) (h_partial_sum : partial_sum = 2345) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 175 ∧ 
  polygon_angle_sum p.n = partial_sum + a + b := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angles_sum_l3281_328121


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l3281_328169

theorem no_solution_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 5 ∧ (3 * x / (x - 5) + 15 / (5 - x) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l3281_328169


namespace NUMINAMATH_CALUDE_quadratic_theorem_l3281_328163

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The maximum value of f occurs at x = 2 -/
def has_max_at_2 (a b c : ℝ) : Prop :=
  ∀ x, f a b c x ≤ f a b c 2

/-- The maximum value of f is 7 -/
def max_value_is_7 (a b c : ℝ) : Prop :=
  f a b c 2 = 7

/-- f passes through the point (0, -7) -/
def passes_through_0_neg7 (a b c : ℝ) : Prop :=
  f a b c 0 = -7

theorem quadratic_theorem (a b c : ℝ) 
  (h1 : has_max_at_2 a b c)
  (h2 : max_value_is_7 a b c)
  (h3 : passes_through_0_neg7 a b c) :
  f a b c 5 = -24.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_theorem_l3281_328163


namespace NUMINAMATH_CALUDE_mod_inverse_sum_l3281_328156

theorem mod_inverse_sum : ∃ (a b : ℤ), 
  (5 * a) % 35 = 1 ∧ 
  (15 * b) % 35 = 1 ∧ 
  (a + b) % 35 = 21 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_l3281_328156


namespace NUMINAMATH_CALUDE_second_apartment_rent_l3281_328166

/-- Calculates the total monthly cost for an apartment --/
def total_monthly_cost (rent : ℚ) (utilities : ℚ) (miles_per_day : ℚ) (work_days : ℚ) (cost_per_mile : ℚ) : ℚ :=
  rent + utilities + (miles_per_day * work_days * cost_per_mile)

/-- The problem statement --/
theorem second_apartment_rent :
  let first_rent : ℚ := 800
  let first_utilities : ℚ := 260
  let first_miles : ℚ := 31
  let second_utilities : ℚ := 200
  let second_miles : ℚ := 21
  let work_days : ℚ := 20
  let cost_per_mile : ℚ := 58 / 100
  let cost_difference : ℚ := 76
  ∃ second_rent : ℚ,
    second_rent = 900 ∧
    total_monthly_cost first_rent first_utilities first_miles work_days cost_per_mile -
    total_monthly_cost second_rent second_utilities second_miles work_days cost_per_mile = cost_difference :=
by
  sorry


end NUMINAMATH_CALUDE_second_apartment_rent_l3281_328166


namespace NUMINAMATH_CALUDE_min_tangent_equals_radius_l3281_328147

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def LineOfSymmetry (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- Point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- Tangent from a point to the circle -/
def Tangent (p : Point) (x y : ℝ) : ℝ :=
  sorry

/-- The radius of the circle -/
def Radius : ℝ :=
  2

theorem min_tangent_equals_radius (a b : ℝ) :
  ∀ (p : Point), p.a = a ∧ p.b = b →
  (∀ (x y : ℝ), Circle x y → LineOfSymmetry a b x y) →
  (∃ (x y : ℝ), Tangent p x y = Radius) ∧
  (∀ (x y : ℝ), Tangent p x y ≥ Radius) :=
sorry

end NUMINAMATH_CALUDE_min_tangent_equals_radius_l3281_328147


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l3281_328198

theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = -1/2 ∧ Q.2 = -Real.sqrt 3/2) ↔ 
  (∃ θ : ℝ, θ = -2*Real.pi/3 ∧ Q.1 = Real.cos θ ∧ Q.2 = Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l3281_328198


namespace NUMINAMATH_CALUDE_ticket_difference_l3281_328115

theorem ticket_difference (fair_tickets : ℕ) (baseball_tickets : ℕ)
  (h1 : fair_tickets = 25)
  (h2 : baseball_tickets = 56) :
  2 * baseball_tickets - fair_tickets = 87 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l3281_328115


namespace NUMINAMATH_CALUDE_ninth_term_value_l3281_328113

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧  -- Definition of arithmetic sequence
  (a 5 + a 7 = 16) ∧                         -- Given condition
  (a 3 = 4)                                  -- Given condition

/-- Theorem stating the value of the 9th term -/
theorem ninth_term_value (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3281_328113
