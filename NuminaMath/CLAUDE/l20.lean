import Mathlib

namespace unfixable_percentage_l20_2038

def total_computers : ℕ := 20
def waiting_percentage : ℚ := 40 / 100
def fixed_right_away : ℕ := 8

theorem unfixable_percentage :
  (total_computers - (waiting_percentage * total_computers).num - fixed_right_away) / total_computers * 100 = 20 := by
  sorry

end unfixable_percentage_l20_2038


namespace sin_pi_eight_squared_l20_2087

theorem sin_pi_eight_squared : 1 - 2 * Real.sin (π / 8) ^ 2 = Real.sqrt 2 / 2 := by sorry

end sin_pi_eight_squared_l20_2087


namespace distinct_real_numbers_inequality_l20_2056

theorem distinct_real_numbers_inequality (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z)
  (eq1 : x^2 - x = y*z)
  (eq2 : y^2 - y = z*x)
  (eq3 : z^2 - z = x*y) :
  -1/3 < x ∧ x < 1 ∧ -1/3 < y ∧ y < 1 ∧ -1/3 < z ∧ z < 1 := by
sorry

end distinct_real_numbers_inequality_l20_2056


namespace jakes_motorcycle_purchase_l20_2061

theorem jakes_motorcycle_purchase (initial_amount : ℝ) (motorcycle_cost : ℝ) (final_amount : ℝ) :
  initial_amount = 5000 ∧
  final_amount = 825 ∧
  final_amount = (initial_amount - motorcycle_cost) / 2 * 3 / 4 →
  motorcycle_cost = 2800 := by
sorry

end jakes_motorcycle_purchase_l20_2061


namespace sequence_difference_l20_2017

theorem sequence_difference (p q : ℕ+) (h : p - q = 5) :
  let S : ℕ+ → ℤ := λ n => 2 * n.val ^ 2 - 3 * n.val
  let a : ℕ+ → ℤ := λ n => S n - if n = 1 then 0 else S (n - 1)
  a p - a q = 20 := by
  sorry

end sequence_difference_l20_2017


namespace train_speed_is_60_mph_l20_2028

/-- The speed of a train given its length and the time it takes to pass another train --/
def train_speed (train_length : ℚ) (passing_time : ℚ) : ℚ :=
  (2 * train_length) / (passing_time / 3600)

/-- Theorem stating that the speed of each train is 60 mph --/
theorem train_speed_is_60_mph (train_length : ℚ) (passing_time : ℚ)
  (h1 : train_length = 1/6)
  (h2 : passing_time = 10) :
  train_speed train_length passing_time = 60 := by
  sorry

#eval train_speed (1/6) 10

end train_speed_is_60_mph_l20_2028


namespace largest_927_triple_l20_2053

/-- Converts a base 10 number to its base 9 representation as a list of digits -/
def toBase9 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Interprets a list of digits as a base 10 number -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is a 9-27 triple -/
def is927Triple (n : ℕ) : Prop :=
  fromDigits (toBase9 n) = 3 * n

/-- States that 108 is the largest 9-27 triple -/
theorem largest_927_triple :
  (∀ m : ℕ, m > 108 → ¬(is927Triple m)) ∧ is927Triple 108 := by
  sorry

end largest_927_triple_l20_2053


namespace martin_wasted_time_l20_2054

def traffic_time : ℝ := 2
def freeway_time_multiplier : ℝ := 4

theorem martin_wasted_time : 
  traffic_time + freeway_time_multiplier * traffic_time = 10 := by
  sorry

end martin_wasted_time_l20_2054


namespace no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l20_2094

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) = 180

-- Define triangle types
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∧ t.angles 1 = t.angles 2

def Triangle.isRight (t : Triangle) : Prop :=
  t.angles 0 = 90 ∨ t.angles 1 = 90 ∨ t.angles 2 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angles 0 > 90 ∨ t.angles 1 > 90 ∨ t.angles 2 > 90

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isRight) :=
sorry

-- Theorem stating that equilateral obtuse triangles cannot exist
theorem no_equilateral_obtuse_triangle :
  ∀ t : Triangle, ¬(t.isEquilateral ∧ t.isObtuse) :=
sorry

end no_equilateral_right_triangle_no_equilateral_obtuse_triangle_l20_2094


namespace evaluate_fraction_at_negative_three_l20_2084

theorem evaluate_fraction_at_negative_three :
  let x : ℚ := -3
  (5 + 2 * x * (x + 5) - 5^2) / (2 * x - 5 + 2 * x^3) = 32 / 65 := by
sorry

end evaluate_fraction_at_negative_three_l20_2084


namespace largest_inscribed_circle_radius_for_specific_quadrilateral_l20_2024

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that for a quadrilateral with side lengths 10, 12, 8, and 14,
    the radius of the largest inscribed circle is √24.75 -/
theorem largest_inscribed_circle_radius_for_specific_quadrilateral :
  let q : Quadrilateral := ⟨10, 12, 8, 14⟩
  largest_inscribed_circle_radius q = Real.sqrt 24.75 := by sorry

end largest_inscribed_circle_radius_for_specific_quadrilateral_l20_2024


namespace purely_imaginary_condition_l20_2091

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- For a real number a, (a+i)(1+2i) is purely imaginary if and only if a = 2. -/
theorem purely_imaginary_condition (a : ℝ) : 
  PurelyImaginary ((a : ℂ) + I * (1 + 2*I)) ↔ a = 2 := by
  sorry

end purely_imaginary_condition_l20_2091


namespace solutions_to_z_fourth_equals_16_l20_2068

theorem solutions_to_z_fourth_equals_16 : 
  {z : ℂ | z^4 = 16} = {2, -2, 2*I, -2*I} := by sorry

end solutions_to_z_fourth_equals_16_l20_2068


namespace partition_existence_l20_2030

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1) ∧ 0 < a n

/-- A partition of ℕ into infinitely many subsets -/
def Partition (A : ℕ → Set ℕ) : Prop :=
  (∀ i j : ℕ, i ≠ j → A i ∩ A j = ∅) ∧
  (∀ n : ℕ, ∃ i : ℕ, n ∈ A i) ∧
  (∀ i : ℕ, Set.Infinite (A i))

/-- The condition on consecutive elements in each subset -/
def SatisfiesCondition (A : ℕ → Set ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i k : ℕ, ∀ b : ℕ → ℕ,
    (∀ n : ℕ, b n ∈ A i ∧ b n < b (n + 1)) →
    (∀ n : ℕ, n + 1 ≤ a k → b (n + 1) - b n ≤ k)

theorem partition_existence :
  ∀ a : ℕ → ℕ, StrictlyIncreasingSeq a →
  ∃ A : ℕ → Set ℕ, Partition A ∧ SatisfiesCondition A a :=
sorry

end partition_existence_l20_2030


namespace shrimp_earnings_l20_2008

theorem shrimp_earnings (victor_shrimp : ℕ) (austin_less : ℕ) (price : ℚ) (tails_per_set : ℕ) : 
  victor_shrimp = 26 →
  austin_less = 8 →
  price = 7 →
  tails_per_set = 11 →
  let austin_shrimp := victor_shrimp - austin_less
  let total_victor_austin := victor_shrimp + austin_shrimp
  let brian_shrimp := total_victor_austin / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let sets_sold := total_shrimp / tails_per_set
  let total_earnings := price * (sets_sold : ℚ)
  let each_boy_earnings := total_earnings / 3
  each_boy_earnings = 14 := by
sorry

end shrimp_earnings_l20_2008


namespace last_integer_is_768_l20_2050

/-- A sequence of 10 distinct positive integers where each (except the first) is a multiple of the previous one -/
def IntegerSequence : Type := Fin 10 → ℕ+

/-- The property that each integer (except the first) is a multiple of the previous one -/
def IsMultipleSequence (seq : IntegerSequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℕ+, seq (i.succ) = k * seq i

/-- The property that all integers in the sequence are distinct -/
def IsDistinct (seq : IntegerSequence) : Prop :=
  ∀ i j : Fin 10, i ≠ j → seq i ≠ seq j

/-- The last integer is between 600 and 1000 -/
def LastIntegerInRange (seq : IntegerSequence) : Prop :=
  600 < seq 9 ∧ seq 9 < 1000

theorem last_integer_is_768 (seq : IntegerSequence) 
  (h1 : IsMultipleSequence seq) 
  (h2 : IsDistinct seq) 
  (h3 : LastIntegerInRange seq) : 
  seq 9 = 768 := by
  sorry

end last_integer_is_768_l20_2050


namespace total_earnings_l20_2083

def working_game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

theorem total_earnings : List.sum working_game_prices = 68 := by
  sorry

end total_earnings_l20_2083


namespace square_areas_side_lengths_sum_l20_2051

theorem square_areas_side_lengths_sum (r1 r2 r3 : ℚ) 
  (h_ratio : r1 = 345/45 ∧ r2 = 345/30 ∧ r3 = 345/15) :
  ∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℕ),
    (r1.sqrt = (a1 : ℚ) * (b1 : ℚ).sqrt / c1) ∧
    (r2.sqrt = (a2 : ℚ) * (b2 : ℚ).sqrt / c2) ∧
    (r3.sqrt = (a3 : ℚ) * (b3 : ℚ).sqrt / c3) ∧
    (a1 + b1 + c1 = 73) ∧
    (a2 + b2 + c2 = 49) ∧
    (a3 + b3 + c3 = 531) ∧
    (max (a1 + b1 + c1) (max (a2 + b2 + c2) (a3 + b3 + c3)) = 531) :=
by sorry

end square_areas_side_lengths_sum_l20_2051


namespace white_balls_count_l20_2003

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) (h_total : total = 32) (h_p_yellow : p_yellow = 1/4) :
  total - (total * p_yellow).floor = 24 :=
sorry

end white_balls_count_l20_2003


namespace polar_to_circle_l20_2010

/-- The polar equation r = 1 / (1 - sin θ) represents a circle. -/
theorem polar_to_circle : ∃ (h k R : ℝ), ∀ (x y : ℝ),
  (∃ (r θ : ℝ), r = 1 / (1 - Real.sin θ) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  (x - h)^2 + (y - k)^2 = R^2 :=
by sorry

end polar_to_circle_l20_2010


namespace range_of_x_range_of_m_l20_2043

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 6*x + 8 < 0
def q (x m : ℝ) : Prop := m - 2 < x ∧ x < m + 1

-- Theorem for the range of x when p is true
theorem range_of_x : Set.Ioo 2 4 = {x : ℝ | p x} := by sorry

-- Theorem for the range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∃ m, q x m) → 
  Set.Icc 3 4 = {m : ℝ | ∀ x, p x → q x m} := by sorry

end range_of_x_range_of_m_l20_2043


namespace part_one_part_two_l20_2057

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x + 1|

-- Part I
theorem part_one :
  {x : ℝ | f 1 x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
sorry

-- Part II
theorem part_two :
  {a : ℝ | ∃ x ≥ a, f a x ≤ 2*a + x} = {a : ℝ | a ≥ 1} :=
sorry

end part_one_part_two_l20_2057


namespace solution_volume_proof_l20_2078

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 440

/-- The percentage of water in the initial solution -/
def initial_water_percent : ℝ := 88

/-- The percentage of concentrated kola in the initial solution -/
def initial_kola_percent : ℝ := 8

/-- The volume of sugar added in liters -/
def added_sugar : ℝ := 3.2

/-- The volume of water added in liters -/
def added_water : ℝ := 10

/-- The volume of concentrated kola added in liters -/
def added_kola : ℝ := 6.8

/-- The percentage of sugar in the final solution -/
def final_sugar_percent : ℝ := 4.521739130434784

theorem solution_volume_proof :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_sugar := initial_volume * initial_sugar_percent / 100
  let final_volume := initial_volume + added_sugar + added_water + added_kola
  let final_sugar := initial_sugar + added_sugar
  (final_sugar / final_volume) * 100 = final_sugar_percent :=
by sorry

end solution_volume_proof_l20_2078


namespace stratified_sampling_theorem_city_stratified_sampling_l20_2005

/-- Represents the types of schools in the city -/
inductive SchoolType
  | University
  | MiddleSchool
  | PrimarySchool

/-- Represents the distribution of schools in the city -/
structure SchoolDistribution where
  total : ℕ
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Represents the sample size and distribution in stratified sampling -/
structure StratifiedSample where
  sampleSize : ℕ
  universitiesSample : ℕ
  middleSchoolsSample : ℕ
  primarySchoolsSample : ℕ

def citySchools : SchoolDistribution :=
  { total := 500
  , universities := 10
  , middleSchools := 200
  , primarySchools := 290 }

def sampleSize : ℕ := 50

theorem stratified_sampling_theorem (d : SchoolDistribution) (s : ℕ) :
  d.total = d.universities + d.middleSchools + d.primarySchools →
  s ≤ d.total →
  ∃ (sample : StratifiedSample),
    sample.sampleSize = s ∧
    sample.universitiesSample = (s * d.universities) / d.total ∧
    sample.middleSchoolsSample = (s * d.middleSchools) / d.total ∧
    sample.primarySchoolsSample = (s * d.primarySchools) / d.total ∧
    sample.sampleSize = sample.universitiesSample + sample.middleSchoolsSample + sample.primarySchoolsSample :=
by sorry

theorem city_stratified_sampling :
  ∃ (sample : StratifiedSample),
    sample.sampleSize = sampleSize ∧
    sample.universitiesSample = 1 ∧
    sample.middleSchoolsSample = 20 ∧
    sample.primarySchoolsSample = 29 :=
by sorry

end stratified_sampling_theorem_city_stratified_sampling_l20_2005


namespace negation_of_forall_positive_negation_of_squared_plus_one_positive_l20_2099

theorem negation_of_forall_positive (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_squared_plus_one_positive :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_squared_plus_one_positive_l20_2099


namespace casino_table_ratio_l20_2007

/-- Proves that the ratio of money on table B to table C is 2 given the casino table conditions -/
theorem casino_table_ratio : 
  ∀ (A B C : ℝ),
  A = 40 →
  C = A + 20 →
  A + B + C = 220 →
  B / C = 2 := by
sorry

end casino_table_ratio_l20_2007


namespace fish_thrown_back_l20_2070

theorem fish_thrown_back (morning_catch : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) (total_catch : ℕ) 
  (h1 : morning_catch = 8)
  (h2 : afternoon_catch = 5)
  (h3 : dad_catch = 13)
  (h4 : total_catch = 23)
  (h5 : total_catch = morning_catch - thrown_back + afternoon_catch + dad_catch) :
  thrown_back = 3 := by
  sorry

end fish_thrown_back_l20_2070


namespace four_m_squared_minus_n_squared_l20_2035

theorem four_m_squared_minus_n_squared (m n : ℝ) 
  (h1 : 2*m + n = 3) (h2 : 2*m - n = 1) : 4*m^2 - n^2 = 3 := by
  sorry

end four_m_squared_minus_n_squared_l20_2035


namespace dakota_bill_is_12190_l20_2092

/-- Calculates Dakota's total medical bill based on given conditions -/
def dakota_medical_bill (
  days : ℕ)
  (bed_charge : ℝ)
  (specialist_rate : ℝ)
  (specialist_time : ℝ)
  (num_specialists : ℕ)
  (ambulance_charge : ℝ)
  (surgery_duration : ℝ)
  (surgeon_rate : ℝ)
  (assistant_rate : ℝ)
  (therapy_rate : ℝ)
  (therapy_duration : ℝ)
  (med_a_cost : ℝ)
  (med_b_cost : ℝ)
  (med_c_rate : ℝ)
  (med_c_duration : ℝ)
  (pills_per_day : ℕ) : ℝ :=
  let bed_total := days * bed_charge
  let specialist_total := days * specialist_rate * specialist_time * num_specialists
  let surgery_total := surgery_duration * (surgeon_rate + assistant_rate)
  let therapy_total := days * therapy_rate * therapy_duration
  let med_a_total := days * med_a_cost * pills_per_day
  let med_b_total := days * med_b_cost * pills_per_day
  let med_c_total := days * med_c_rate * med_c_duration
  bed_total + specialist_total + ambulance_charge + surgery_total + therapy_total + med_a_total + med_b_total + med_c_total

/-- Theorem stating that Dakota's medical bill is $12,190 -/
theorem dakota_bill_is_12190 :
  dakota_medical_bill 3 900 250 0.25 2 1800 2 1500 800 300 1 20 45 80 2 3 = 12190 := by
  sorry

end dakota_bill_is_12190_l20_2092


namespace train_passing_time_l20_2075

/-- Time for a train to pass a moving platform -/
theorem train_passing_time (train_length platform_length : ℝ) 
  (train_speed platform_speed : ℝ) : 
  train_length = 157 →
  platform_length = 283 →
  train_speed = 72 →
  platform_speed = 18 →
  (train_length + platform_length) / ((train_speed - platform_speed) * (1000 / 3600)) = 
    440 / (54 * (1000 / 3600)) := by
  sorry

end train_passing_time_l20_2075


namespace equation_has_three_solutions_l20_2052

-- Define the equation
def f (x : ℝ) : Prop := Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (f a ∧ f b ∧ f c) ∧
  (∀ x : ℝ, f x → (x = a ∨ x = b ∨ x = c)) :=
sorry

end equation_has_three_solutions_l20_2052


namespace triangle_proof_l20_2019

theorem triangle_proof (a b c : ℝ) (ha : a = 18) (hb : b = 24) (hc : c = 30) :
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧  -- Triangle inequality
  (a^2 + b^2 = c^2) ∧                    -- Right triangle
  (1/2 * a * b = 216) :=                 -- Area
by sorry

end triangle_proof_l20_2019


namespace figure_can_form_square_l20_2067

/-- Represents a point on a 2D plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle on a 2D plane --/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the original figure --/
def OriginalFigure : Type := List Point

/-- Represents a square --/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Function to cut the original figure into 5 triangles --/
def cutIntoTriangles (figure : OriginalFigure) : List Triangle := sorry

/-- Function to check if a list of triangles can form a square --/
def canFormSquare (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that the original figure can be cut into 5 triangles and rearranged to form a square --/
theorem figure_can_form_square (figure : OriginalFigure) : 
  ∃ (triangles : List Triangle), 
    triangles = cutIntoTriangles figure ∧ 
    triangles.length = 5 ∧ 
    canFormSquare triangles := sorry

end figure_can_form_square_l20_2067


namespace percent_relation_l20_2002

theorem percent_relation (P Q : ℝ) (h : (1/2) * P = (1/5) * Q) :
  P = (2/5) * Q := by
  sorry

end percent_relation_l20_2002


namespace governors_addresses_l20_2025

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem governors_addresses : total_addresses = 40 := by
  sorry

end governors_addresses_l20_2025


namespace complex_magnitude_problem_l20_2082

theorem complex_magnitude_problem (z : ℂ) : z = (1 - I) / (1 + I) + 2*I → Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l20_2082


namespace function_passes_through_point_one_one_l20_2012

/-- The function f(x) = a^(x-1) always passes through the point (1, 1) for any a > 0 and a ≠ 1 -/
theorem function_passes_through_point_one_one (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end function_passes_through_point_one_one_l20_2012


namespace inequality_solution_set_l20_2004

open Real

theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Ioo (-1 : ℝ) 1 →
  (abs (sin x) + abs (log (1 - x^2)) > abs (sin x + log (1 - x^2))) ↔ x ∈ Set.Ioo 0 1 := by
  sorry

end inequality_solution_set_l20_2004


namespace brads_running_speed_l20_2039

theorem brads_running_speed
  (distance_between_homes : ℝ)
  (maxwells_speed : ℝ)
  (time_until_meeting : ℝ)
  (brads_delay : ℝ)
  (h1 : distance_between_homes = 54)
  (h2 : maxwells_speed = 4)
  (h3 : time_until_meeting = 6)
  (h4 : brads_delay = 1)
  : ∃ (brads_speed : ℝ), brads_speed = 6 := by
  sorry

end brads_running_speed_l20_2039


namespace tom_needs_163_blue_tickets_l20_2093

/-- Represents the number of tickets Tom has -/
structure TomTickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the number of additional blue tickets needed to win the Bible -/
def additional_blue_tickets_needed (tickets : TomTickets) : ℕ :=
  let yellow_to_blue := 100
  let red_to_blue := 10
  let total_blue_needed := 10 * yellow_to_blue
  let blue_from_yellow := tickets.yellow * yellow_to_blue
  let blue_from_red := tickets.red * red_to_blue
  let blue_total := blue_from_yellow + blue_from_red + tickets.blue
  total_blue_needed - blue_total

/-- Theorem stating that Tom needs 163 more blue tickets to win the Bible -/
theorem tom_needs_163_blue_tickets :
  additional_blue_tickets_needed ⟨8, 3, 7⟩ = 163 := by
  sorry


end tom_needs_163_blue_tickets_l20_2093


namespace triangle_problem_l20_2065

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that B = π/3 and AD = (2√13)/3 under certain conditions. -/
theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real) :
  0 < A ∧ A < π/2 →  -- Triangle ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.sin A = a * Real.cos (B - π/6) →  -- Given condition
  b = Real.sqrt 13 →  -- Given condition
  a = 4 →  -- Given condition
  0 ≤ D ∧ D ≤ c →  -- D is on AC
  (1/2) * a * D * Real.sin B = 2 * Real.sqrt 3 →  -- Area of ABD
  B = π/3 ∧ D = (2 * Real.sqrt 13) / 3 := by
  sorry


end triangle_problem_l20_2065


namespace action_figures_per_shelf_l20_2081

theorem action_figures_per_shelf 
  (total_shelves : ℕ) 
  (total_figures : ℕ) 
  (h1 : total_shelves = 3) 
  (h2 : total_figures = 27) : 
  total_figures / total_shelves = 9 := by
  sorry

end action_figures_per_shelf_l20_2081


namespace min_value_of_function_min_value_achieved_l20_2073

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + (1 / (x - 1)) = 3 ↔ x = 2 :=
sorry

end min_value_of_function_min_value_achieved_l20_2073


namespace geometric_sequence_sum_l20_2042

/-- Given a geometric sequence {aₙ} where a₁ + a₂ + a₃ = 1 and a₂ + a₃ + a₄ = 2,
    prove that a₈ + a₉ + a₁₀ = 128 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 + a 3 = 1 →                      -- First condition
  a 2 + a 3 + a 4 = 2 →                      -- Second condition
  a 8 + a 9 + a 10 = 128 :=                  -- Conclusion to prove
by sorry

end geometric_sequence_sum_l20_2042


namespace multiplication_puzzle_l20_2086

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → (30 + c) * (10 * d + 4) = 146 → c + d = 7 := by
  sorry

end multiplication_puzzle_l20_2086


namespace race_catch_up_time_l20_2014

/-- Proves that Nicky runs for 30 seconds before Cristina catches up in a 300-meter race --/
theorem race_catch_up_time 
  (race_distance : ℝ) 
  (head_start : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (h1 : race_distance = 300)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) : 
  ∃ (t : ℝ), t = 30 ∧ 
  cristina_speed * (t - head_start) = nicky_speed * t := by
  sorry


end race_catch_up_time_l20_2014


namespace circle_with_chords_theorem_l20_2009

/-- Represents a circle with two intersecting chords --/
structure CircleWithChords where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form mπ - n√d --/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) --/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n) → p = 1

/-- Main theorem about the circle with intersecting chords --/
theorem circle_with_chords_theorem (circle : CircleWithChords) 
  (h1 : circle.radius = 36)
  (h2 : circle.chord_length = 66)
  (h3 : circle.intersection_distance = 12) :
  ∃ (area : RegionArea), 
    (area.m : ℝ) * Real.pi - (area.n : ℝ) * Real.sqrt (area.d : ℝ) > 0 ∧
    is_square_free area.d ∧
    area.m + area.n + area.d = 378 :=
  sorry

end circle_with_chords_theorem_l20_2009


namespace sum_of_single_digit_numbers_l20_2074

theorem sum_of_single_digit_numbers (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → B = A - 2 → A = 5 + 3 → A + B = 14 := by
  sorry

end sum_of_single_digit_numbers_l20_2074


namespace max_sum_of_digits_24hour_l20_2066

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_24hour : (⨆ t : Time24, sumOfDigitsTime24 t) = 24 := by
  sorry

end max_sum_of_digits_24hour_l20_2066


namespace white_surface_fraction_is_11_16_l20_2044

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℝ
  edge_positive : edge_length > 0

/-- Represents the larger cube constructed from smaller cubes -/
structure LargeCube where
  cube : Cube
  small_cubes : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  black_corners : ℕ
  black_face_centers : ℕ

/-- Calculates the fraction of white surface area for the large cube -/
def white_surface_fraction (lc : LargeCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem white_surface_fraction_is_11_16 :
  let lc : LargeCube := {
    cube := { edge_length := 4, edge_positive := by norm_num },
    small_cubes := 64,
    black_cubes := 24,
    white_cubes := 40,
    black_corners := 8,
    black_face_centers := 6
  }
  white_surface_fraction lc = 11 / 16 := by
  sorry

end white_surface_fraction_is_11_16_l20_2044


namespace tangent_line_constant_l20_2089

theorem tangent_line_constant (a k b : ℝ) : 
  (∀ x, a * x^2 + 2 + Real.log x = k * x + b) →  -- The line is tangent to the curve
  (1 : ℝ)^2 * a + 2 + Real.log 1 = k * 1 + b →   -- The point (1, 4) lies on both the line and curve
  (4 : ℝ) = k * 1 + b →                          -- The y-coordinate of P is 4
  (∀ x, 2 * a * x + 1 / x = k) →                 -- The derivatives are equal at x = 1
  b = -1 := by
sorry


end tangent_line_constant_l20_2089


namespace perfect_square_trinomial_l20_2080

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + m*x + 16 = (a*x + b)^2) → (m = 8 ∨ m = -8) := by
  sorry

end perfect_square_trinomial_l20_2080


namespace cricket_average_increase_l20_2022

theorem cricket_average_increase 
  (innings : ℕ) 
  (current_average : ℚ) 
  (next_innings_score : ℕ) 
  (average_increase : ℚ) : 
  innings = 20 → 
  current_average = 36 → 
  next_innings_score = 120 → 
  (innings : ℚ) * current_average + next_innings_score = (innings + 1) * (current_average + average_increase) → 
  average_increase = 4 := by
sorry

end cricket_average_increase_l20_2022


namespace climb_down_distance_is_6_l20_2046

-- Define the climb up speed
def climb_up_speed : ℝ := 2

-- Define the climb down speed
def climb_down_speed : ℝ := 3

-- Define the total time
def total_time : ℝ := 4

-- Define the additional distance on the way down
def additional_distance : ℝ := 2

-- Theorem statement
theorem climb_down_distance_is_6 :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x / climb_up_speed + (x + additional_distance) / climb_down_speed = total_time ∧
    x + additional_distance = 6 :=
sorry

end climb_down_distance_is_6_l20_2046


namespace smallest_three_digit_divisible_by_4_and_5_l20_2016

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 4 = 0 ∧ n % 5 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 4 = 0 ∧ m % 5 = 0 → m ≥ n) ∧
  n = 100 := by
  sorry

end smallest_three_digit_divisible_by_4_and_5_l20_2016


namespace triangle_perimeter_bound_l20_2055

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, s > 0 → 7 + s > 25 → 25 + s > 7 → 7 + 25 + s < 65 := by
  sorry

end triangle_perimeter_bound_l20_2055


namespace work_completion_days_l20_2062

/-- Given a number of people P and the number of days D it takes them to complete a work,
    prove that D = 4 when double the number of people can do half the work in 2 days. -/
theorem work_completion_days (P : ℕ) (D : ℕ) (h : P * D = 2 * P * 2 * 2) : D = 4 := by
  sorry

end work_completion_days_l20_2062


namespace min_value_of_expression_l20_2079

theorem min_value_of_expression (a b : ℤ) (h1 : a > b) (h2 : a ≠ b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a b : ℤ), a > b ∧ a ≠ b ∧ (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) = 2 :=
by sorry

end min_value_of_expression_l20_2079


namespace incircle_segment_ratio_l20_2088

/-- Represents a triangle with an incircle -/
structure TriangleWithIncircle where
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  r : ℝ  -- Smaller segment of side 'a' created by incircle
  s : ℝ  -- Larger segment of side 'a' created by incircle
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : r + s = a
  h_r_lt_s : r < s

/-- The main theorem -/
theorem incircle_segment_ratio
  (t : TriangleWithIncircle)
  (h_side_lengths : t.a = 8 ∧ t.b = 13 ∧ t.c = 17) :
  t.r / t.s = 1 / 3 := by
  sorry

end incircle_segment_ratio_l20_2088


namespace systematic_sample_fourth_element_l20_2031

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : Nat
  sampleSize : Nat
  knownSamples : List Nat

/-- Calculates the sampling interval for a systematic sample -/
def samplingInterval (s : SystematicSample) : Nat :=
  s.populationSize / s.sampleSize

/-- Checks if a given number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = (s.knownSamples.head!) + k * samplingInterval s

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.populationSize = 56)
  (h2 : s.sampleSize = 4)
  (h3 : s.knownSamples = [6, 34, 48]) :
  isInSample s 20 := by
  sorry

end systematic_sample_fourth_element_l20_2031


namespace square_sum_ge_twice_product_l20_2023

theorem square_sum_ge_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end square_sum_ge_twice_product_l20_2023


namespace elsa_remaining_data_l20_2090

/-- Calculates the remaining data after Elsa's usage -/
def remaining_data (total : ℚ) (youtube : ℚ) (facebook_fraction : ℚ) : ℚ :=
  let after_youtube := total - youtube
  let facebook_usage := facebook_fraction * after_youtube
  after_youtube - facebook_usage

/-- Theorem stating that Elsa's remaining data is 120 MB -/
theorem elsa_remaining_data :
  remaining_data 500 300 (2/5) = 120 := by
  sorry

#eval remaining_data 500 300 (2/5)

end elsa_remaining_data_l20_2090


namespace pythagorean_theorem_3_4_5_l20_2045

theorem pythagorean_theorem_3_4_5 :
  let a : ℝ := 30
  let b : ℝ := 40
  let c : ℝ := 50
  a^2 + b^2 = c^2 := by sorry

end pythagorean_theorem_3_4_5_l20_2045


namespace two_digit_number_interchange_l20_2063

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 3 → 
  (10 * x + y) - (10 * y + x) = 27 := by
sorry

end two_digit_number_interchange_l20_2063


namespace min_distance_sum_l20_2064

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle F
def F (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define a line passing through F(0,1)
def line_through_F (m : ℝ) (x y : ℝ) : Prop := x = m*(y-1)

-- Define the theorem
theorem min_distance_sum (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    line_through_F m x1 y1 ∧ line_through_F m x2 y2 ∧
    y1 + 2*y2 ≥ 2*Real.sqrt 2 :=
sorry

end min_distance_sum_l20_2064


namespace library_visitors_average_average_visitors_proof_l20_2076

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 540) (h2 : other_day_visitors = 240) : ℕ :=
let total_sundays := 5
let total_other_days := 25
let total_days := 30
let total_visitors := sunday_visitors * total_sundays + other_day_visitors * total_other_days
let average_visitors := total_visitors / total_days
290

theorem average_visitors_proof (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 540) (h2 : other_day_visitors = 240) :
  library_visitors_average sunday_visitors other_day_visitors h1 h2 = 290 := by
sorry

end library_visitors_average_average_visitors_proof_l20_2076


namespace equidistant_point_y_axis_l20_2026

theorem equidistant_point_y_axis (y : ℝ) : 
  (∀ (x : ℝ), x = 0 → 
    (x - 3)^2 + y^2 = (x - 5)^2 + (y - 6)^2) → 
  y = 13/3 := by sorry

end equidistant_point_y_axis_l20_2026


namespace olympiad_score_problem_l20_2072

theorem olympiad_score_problem :
  ∀ (x y : ℕ),
    x + y = 14 →
    7 * x - 12 * y = 60 →
    x = 12 :=
by
  sorry

end olympiad_score_problem_l20_2072


namespace not_divisible_by_97_l20_2095

theorem not_divisible_by_97 (k : ℤ) : (99^3 - 99) % k = 0 → k ≠ 97 := by
  sorry

end not_divisible_by_97_l20_2095


namespace duck_cow_problem_l20_2060

theorem duck_cow_problem (D C : ℕ) : 
  (2 * D + 4 * C = 2 * (D + C) + 22) → C = 11 := by
  sorry

end duck_cow_problem_l20_2060


namespace first_five_valid_codes_l20_2037

def is_valid_code (n : ℕ) : Bool := n < 800

def extract_codes (seq : List ℕ) : List ℕ :=
  seq.filter is_valid_code

theorem first_five_valid_codes 
  (random_sequence : List ℕ := [785, 916, 955, 567, 199, 981, 050, 717, 512]) :
  (extract_codes random_sequence).take 5 = [785, 567, 199, 507, 175] := by
  sorry

end first_five_valid_codes_l20_2037


namespace infinitely_many_primes_mod_3_eq_2_l20_2001

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end infinitely_many_primes_mod_3_eq_2_l20_2001


namespace complex_equal_parts_l20_2098

theorem complex_equal_parts (a : ℝ) : 
  (Complex.re ((1 + a * Complex.I) * (2 + Complex.I)) = 
   Complex.im ((1 + a * Complex.I) * (2 + Complex.I))) → 
  a = 1/3 := by
sorry

end complex_equal_parts_l20_2098


namespace ones_digit_of_7_pow_35_l20_2032

/-- The ones digit of 7^n -/
def ones_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustive pattern matching

/-- Theorem stating that the ones digit of 7^35 is 3 -/
theorem ones_digit_of_7_pow_35 : ones_digit_of_7_pow 35 = 3 := by
  sorry

#eval ones_digit_of_7_pow 35

end ones_digit_of_7_pow_35_l20_2032


namespace boat_width_proof_l20_2069

theorem boat_width_proof (river_width : ℝ) (num_boats : ℕ) (min_space : ℝ) 
  (h1 : river_width = 42)
  (h2 : num_boats = 8)
  (h3 : min_space = 2)
  (h4 : ∃ boat_width : ℝ, river_width = num_boats * boat_width + (num_boats + 1) * min_space) :
  ∃ boat_width : ℝ, boat_width = 3 := by
sorry

end boat_width_proof_l20_2069


namespace quadratic_solution_difference_l20_2071

theorem quadratic_solution_difference (x : ℝ) : 
  x^2 - 5*x + 15 = x + 35 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 - 5*x1 + 15 = x1 + 35) ∧ (x2^2 - 5*x2 + 15 = x2 + 35) ∧ 
  (max x1 x2 - min x1 x2 = 2 * Real.sqrt 29) :=
by sorry

end quadratic_solution_difference_l20_2071


namespace wire_cut_square_octagon_ratio_l20_2018

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square and b forms a regular octagon with equal perimeters, prove that a/b = 1 -/
theorem wire_cut_square_octagon_ratio (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 4 * (a / 4) = 8 * (b / 8)) : a / b = 1 := by
  sorry

end wire_cut_square_octagon_ratio_l20_2018


namespace smallest_number_l20_2011

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -1) (h3 : c = -Real.sqrt 3) (h4 : d = 3) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_number_l20_2011


namespace positive_real_inequality_l20_2059

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 / y^2) + (y^2 / z^2) + (z^2 / x^2) ≥ (x / y) + (y / z) + (z / x) := by
  sorry

end positive_real_inequality_l20_2059


namespace max_container_weight_for_guaranteed_loading_l20_2077

/-- Represents a valid loading configuration -/
structure LoadingConfig where
  containers : List ℕ
  platforms : List (List ℕ)

/-- Checks if a loading configuration is valid -/
def isValidConfig (total_weight : ℕ) (max_weight : ℕ) (platform_capacity : ℕ) (platform_count : ℕ) (config : LoadingConfig) : Prop :=
  (config.containers.sum = total_weight) ∧
  (∀ c ∈ config.containers, c ≤ max_weight) ∧
  (config.platforms.length = platform_count) ∧
  (∀ p ∈ config.platforms, p.sum ≤ platform_capacity) ∧
  (config.containers.toFinset = config.platforms.join.toFinset)

/-- Theorem stating the maximum container weight for guaranteed loading -/
theorem max_container_weight_for_guaranteed_loading
  (total_weight : ℕ)
  (platform_capacity : ℕ)
  (platform_count : ℕ)
  (h_total : total_weight = 1500)
  (h_capacity : platform_capacity = 80)
  (h_count : platform_count = 25) :
  (∀ k : ℕ, k ≤ 26 →
    ∀ containers : List ℕ,
      (containers.sum = total_weight ∧ ∀ c ∈ containers, c ≤ k) →
      ∃ config : LoadingConfig, isValidConfig total_weight k platform_capacity platform_count config) ∧
  ¬(∀ containers : List ℕ,
      (containers.sum = total_weight ∧ ∀ c ∈ containers, c ≤ 27) →
      ∃ config : LoadingConfig, isValidConfig total_weight 27 platform_capacity platform_count config) :=
by sorry

end max_container_weight_for_guaranteed_loading_l20_2077


namespace sin_geq_cos_range_l20_2041

theorem sin_geq_cos_range (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (Real.sin x ≥ Real.cos x) ↔ (x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end sin_geq_cos_range_l20_2041


namespace divisible_by_seventeen_l20_2034

theorem divisible_by_seventeen (k : ℕ) : 
  17 ∣ (2^(2*k + 3) + 3^(k + 2) * 7^k) := by
  sorry

end divisible_by_seventeen_l20_2034


namespace cos_18_deg_l20_2027

theorem cos_18_deg (h : Real.cos (72 * π / 180) = (Real.sqrt 5 - 1) / 4) :
  Real.cos (18 * π / 180) = Real.sqrt (5 + Real.sqrt 5) / 4 := by
  sorry

end cos_18_deg_l20_2027


namespace binomial_coefficient_7_4_l20_2020

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_coefficient_7_4_l20_2020


namespace floor_power_minus_n_even_l20_2048

theorem floor_power_minus_n_even (n : ℕ+) : 
  ∃ (u : ℝ), u > 0 ∧ ∀ (n : ℕ+), Even (⌊u^(n : ℝ)⌋ - n) :=
by
  -- The proof goes here
  sorry

end floor_power_minus_n_even_l20_2048


namespace distance_between_cities_l20_2040

/-- The distance between two cities given the speeds of two cars and their arrival time difference -/
theorem distance_between_cities (v_slow v_fast : ℝ) (time_diff : ℝ) : 
  v_slow = 72 →
  v_fast = 78 →
  time_diff = 1/3 →
  ∃ d : ℝ, d = 312 ∧ d = v_fast * (d / v_fast) ∧ d = v_slow * (d / v_fast + time_diff) :=
by sorry

end distance_between_cities_l20_2040


namespace isosceles_triangle_perimeter_l20_2049

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  IsoscelesTriangle a b c →
  a + b + c = 30 →
  ((2 * (a + b + c) / 5 = a ∧ (a + b + c) / 5 = b ∧ (a + b + c) / 5 = c) ∨
   (a = 8 ∧ b = 11 ∧ c = 11) ∨
   (a = 8 ∧ b = 8 ∧ c = 14) ∨
   (a = 11 ∧ b = 8 ∧ c = 11) ∨
   (a = 14 ∧ b = 8 ∧ c = 8)) :=
by sorry

end isosceles_triangle_perimeter_l20_2049


namespace complex_powers_sum_l20_2036

theorem complex_powers_sum (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (π / 4)) : 
  z^12 + z⁻¹^12 = -2 ∧ z^6 + z⁻¹^6 = 0 := by
  sorry

end complex_powers_sum_l20_2036


namespace angle_complement_half_supplement_is_zero_l20_2047

theorem angle_complement_half_supplement_is_zero (x : ℝ) :
  (90 - x) = (1/2) * (180 - x) → x = 0 := by
  sorry

end angle_complement_half_supplement_is_zero_l20_2047


namespace lower_average_price_l20_2029

theorem lower_average_price (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check lower_average_price

end lower_average_price_l20_2029


namespace racing_game_cost_l20_2006

/-- Given that Joan spent $9.43 on video games in total and
    purchased a basketball game for $5.2, prove that the
    cost of the racing game is $9.43 - $5.2. -/
theorem racing_game_cost (total_spent : ℝ) (basketball_cost : ℝ)
    (h1 : total_spent = 9.43)
    (h2 : basketball_cost = 5.2) :
    total_spent - basketball_cost = 9.43 - 5.2 := by
  sorry

end racing_game_cost_l20_2006


namespace max_value_of_f_in_interval_l20_2085

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 6

theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 11 ∧ 
  (∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ (x : ℝ), -4 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end max_value_of_f_in_interval_l20_2085


namespace parallel_line_slope_l20_2013

/-- Given a line with equation 3x + 6y = -24, prove that the slope of any parallel line is -1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x + 6 * y = -24) → (slope_of_parallel_line : ℝ) = -1/2 :=
by
  sorry


end parallel_line_slope_l20_2013


namespace power_division_l20_2096

theorem power_division (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end power_division_l20_2096


namespace cricketer_average_score_l20_2015

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (three_match_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_matches = 5)
  (h2 : three_match_avg = 40)
  (h3 : total_avg = 36) :
  (5 * total_avg - 3 * three_match_avg) / 2 = 30 :=
by sorry

end cricketer_average_score_l20_2015


namespace fans_per_bleacher_set_l20_2000

theorem fans_per_bleacher_set (total_fans : ℕ) (num_bleacher_sets : ℕ) 
  (h1 : total_fans = 2436) (h2 : num_bleacher_sets = 3) :
  total_fans / num_bleacher_sets = 812 := by
  sorry

end fans_per_bleacher_set_l20_2000


namespace line_increase_l20_2097

/-- Given a line where an increase of 4 units in x results in an increase of 6 units in y,
    prove that an increase of 12 units in x results in an increase of 18 units in y. -/
theorem line_increase (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 4) - f x = 6) → (f (x + 12) - f x = 18) := by
  sorry

end line_increase_l20_2097


namespace propositions_truth_l20_2033

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Statement of the theorem
theorem propositions_truth : 
  (∀ x > 0, (1/2 : ℝ)^x > (1/3 : ℝ)^x) ∧ 
  (∃ x ∈ Set.Ioo 0 1, log (1/2) x > log (1/3) x) ∧
  (∃ x > 0, (1/2 : ℝ)^x < log (1/2) x) ∧
  (∀ x ∈ Set.Ioo 0 (1/3), (1/2 : ℝ)^x < log (1/3) x) :=
by sorry

end propositions_truth_l20_2033


namespace min_value_expression_l20_2058

theorem min_value_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 2) (hab : a * b = 1) :
  (1 / (2 - a)) + (2 / (2 - b)) ≥ 2 + (2 * Real.sqrt 2) / 3 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ a₀ < 2 ∧ 0 < b₀ ∧ b₀ < 2 ∧ a₀ * b₀ = 1 ∧
    (1 / (2 - a₀)) + (2 / (2 - b₀)) = 2 + (2 * Real.sqrt 2) / 3 :=
by sorry

end min_value_expression_l20_2058


namespace four_digit_number_expansion_l20_2021

/-- Represents a four-digit number with digits a, b, c, and d -/
def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

/-- Theorem stating that a four-digit number with digits a, b, c, and d
    is equal to 1000a + 100b + 10c + d -/
theorem four_digit_number_expansion {a b c d : ℕ} (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : d < 10) :
  four_digit_number a b c d = 1000 * a + 100 * b + 10 * c + d := by
  sorry

end four_digit_number_expansion_l20_2021
