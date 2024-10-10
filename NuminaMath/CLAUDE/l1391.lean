import Mathlib

namespace inequality_proof_l1391_139145

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) ≥ 5/2 ∧
  ((a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) = 5/2 ↔ a = b ∧ b = c) :=
by sorry


end inequality_proof_l1391_139145


namespace equation_one_l1391_139187

theorem equation_one (x : ℝ) : x * (5 * x + 4) = 5 * x + 4 ↔ x = -4/5 ∨ x = 1 := by sorry

end equation_one_l1391_139187


namespace greatest_prime_factor_of_product_l1391_139120

def x : ℕ := 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20

theorem greatest_prime_factor_of_product (x : ℕ) : 
  x = 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20 →
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18 * x * 14 * x) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18 * x * 14 * x) → q ≤ p ∧ p = 7 :=
by sorry

end greatest_prime_factor_of_product_l1391_139120


namespace arithmetic_progression_condition_l1391_139118

def list : List ℤ := [3, 7, 2, 7, 5, 2]

def mean (x : ℚ) : ℚ := (list.sum + x) / 7

def mode : ℤ := 7

noncomputable def median (x : ℚ) : ℚ :=
  if x ≤ 2 then 3
  else if x < 5 then x
  else 5

theorem arithmetic_progression_condition (x : ℚ) :
  (mode : ℚ) < median x ∧ median x < mean x ∧
  median x - mode = mean x - median x →
  x = 75 / 13 := by sorry

end arithmetic_progression_condition_l1391_139118


namespace lines_concurrent_iff_det_zero_l1391_139169

/-- Three lines pass through the same point if and only if the determinant of their coefficients is zero -/
theorem lines_concurrent_iff_det_zero 
  (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : ℝ) : 
  (∃ (x y : ℝ), A₁*x + B₁*y + C₁ = 0 ∧ A₂*x + B₂*y + C₂ = 0 ∧ A₃*x + B₃*y + C₃ = 0) ↔ 
  Matrix.det !![A₁, B₁, C₁; A₂, B₂, C₂; A₃, B₃, C₃] = 0 :=
by sorry

end lines_concurrent_iff_det_zero_l1391_139169


namespace consecutive_odd_integers_sum_l1391_139158

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x % 2 = 1 → -- x is odd
  (x + 4) % 2 = 1 → -- x+4 is odd
  x + (x + 4) = 138 → -- sum of first and third is 138
  x + (x + 2) + (x + 4) = 207 := by
sorry

end consecutive_odd_integers_sum_l1391_139158


namespace flag_distribution_theorem_l1391_139161

/-- Represents the colors of flags -/
inductive FlagColor
  | Blue
  | Red
  | Green

/-- Represents a pair of flags -/
structure FlagPair where
  first : FlagColor
  second : FlagColor

/-- The distribution of flag pairs among children -/
structure FlagDistribution where
  blueRed : ℚ
  redGreen : ℚ
  blueGreen : ℚ
  allThree : ℚ

/-- The problem statement -/
theorem flag_distribution_theorem (dist : FlagDistribution) :
  dist.blueRed = 1/2 →
  dist.redGreen = 3/10 →
  dist.blueGreen = 1/10 →
  dist.allThree = 1/10 →
  dist.blueRed + dist.redGreen + dist.blueGreen + dist.allThree = 1 →
  (dist.blueRed + dist.redGreen + dist.blueGreen - dist.allThree + dist.allThree : ℚ) = 9/10 :=
by sorry

end flag_distribution_theorem_l1391_139161


namespace set_operations_l1391_139134

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x | x > 1}) ∧
  (A ∩ (Set.univ \ B) = {x | 1 < x ∧ x < 2}) := by
  sorry

end set_operations_l1391_139134


namespace tan_alpha_tan_beta_l1391_139150

theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : (Real.cos (α - β))^2 - (Real.cos (α + β))^2 = 1/2)
  (h2 : (1 + Real.cos (2 * α)) * (1 + Real.cos (2 * β)) = 1/3) :
  Real.tan α * Real.tan β = 3/8 := by
  sorry

end tan_alpha_tan_beta_l1391_139150


namespace f_max_min_on_interval_l1391_139126

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 1

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧
  (∀ x ∈ interval, f x ≤ f a) ∧
  (∀ x ∈ interval, f b ≤ f x) ∧
  f a = 1 ∧ f b = -2 :=
sorry

end f_max_min_on_interval_l1391_139126


namespace stevens_peaches_l1391_139162

/-- Given that Jake has 7 peaches and 12 fewer peaches than Steven, prove that Steven has 19 peaches. -/
theorem stevens_peaches (jake_peaches : ℕ) (steven_jake_diff : ℕ) 
  (h1 : jake_peaches = 7)
  (h2 : steven_jake_diff = 12) :
  jake_peaches + steven_jake_diff = 19 := by
sorry

end stevens_peaches_l1391_139162


namespace last_two_digits_of_2005_power_1989_l1391_139190

theorem last_two_digits_of_2005_power_1989 :
  (2005^1989) % 100 = 25 := by
  sorry

end last_two_digits_of_2005_power_1989_l1391_139190


namespace min_distance_to_line_l1391_139199

theorem min_distance_to_line (x y : ℝ) (h : x + y - 3 = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧
  ∀ (x' y' : ℝ), x' + y' - 3 = 0 →
  min ≤ Real.sqrt ((x' - 2)^2 + (y' + 1)^2) :=
sorry

end min_distance_to_line_l1391_139199


namespace trig_identity_l1391_139115

theorem trig_identity : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l1391_139115


namespace smallest_constant_two_l1391_139128

/-- A function satisfying the given conditions on the interval [0,1] -/
structure SpecialFunction where
  f : Real → Real
  domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x
  f_one : f 1 = 1
  subadditive : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ x + y ∧ x + y ≤ 1 → 
    f x + f y ≤ f (x + y)

/-- The theorem stating that 2 is the smallest constant c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_constant_two (sf : SpecialFunction) : 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ 2 * x) ∧ 
  (∀ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ c * x) → 2 ≤ c) :=
sorry

end smallest_constant_two_l1391_139128


namespace committee_selection_ways_l1391_139186

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committee_selection_ways :
  let total_members : ℕ := 30
  let committee_size : ℕ := 5
  choose total_members committee_size = 142506 := by
  sorry

end committee_selection_ways_l1391_139186


namespace abc_maximum_l1391_139194

theorem abc_maximum (a b c : ℝ) (h1 : 2 * a + b = 4) (h2 : a * b + c = 5) :
  ∃ (max : ℝ), ∀ (x y z : ℝ), 2 * x + y = 4 → x * y + z = 5 → x * y * z ≤ max ∧ a * b * c = max :=
by
  sorry

end abc_maximum_l1391_139194


namespace cube_split_and_stack_l1391_139143

/-- The number of millimeters in a meter -/
def mm_per_m : ℕ := 1000

/-- The number of meters in a kilometer -/
def m_per_km : ℕ := 1000

/-- The edge length of the original cube in meters -/
def cube_edge_m : ℕ := 1

/-- The edge length of small cubes in millimeters -/
def small_cube_edge_mm : ℕ := 1

/-- The height of the column in kilometers -/
def column_height_km : ℕ := 1000

theorem cube_split_and_stack :
  (cube_edge_m * mm_per_m)^3 / small_cube_edge_mm = column_height_km * m_per_km * mm_per_m :=
sorry

end cube_split_and_stack_l1391_139143


namespace square_difference_of_integers_l1391_139106

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 70)
  (diff_eq : a - b = 14) :
  a ^ 2 - b ^ 2 = 980 := by
sorry

end square_difference_of_integers_l1391_139106


namespace star_equation_solution_l1391_139192

-- Define the custom operation ※
def star (a b : ℝ) : ℝ := a^2 - 3*a + b

-- State the theorem
theorem star_equation_solution :
  ∃ x₁ x₂ : ℝ, (x₁ = -1 ∨ x₁ = 4) ∧ (x₂ = -1 ∨ x₂ = 4) ∧
  (∀ x : ℝ, star x 2 = 6 ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end star_equation_solution_l1391_139192


namespace smallest_angle_in_right_triangle_l1391_139135

theorem smallest_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → a / b = 5 / 4 → min a b = 40 := by
sorry

end smallest_angle_in_right_triangle_l1391_139135


namespace ends_with_2015_l1391_139189

theorem ends_with_2015 : ∃ n : ℕ, ∃ k : ℕ, 90 * n + 75 = 10000 * k + 2015 := by
  sorry

end ends_with_2015_l1391_139189


namespace correct_height_l1391_139191

theorem correct_height (n : ℕ) (initial_avg : ℝ) (incorrect_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧
  initial_avg = 175 ∧
  incorrect_height = 151 ∧
  actual_avg = 174.5 →
  ∃ (actual_height : ℝ),
    actual_height = 166 ∧
    n * actual_avg = (n - 1) * initial_avg + actual_height - incorrect_height :=
by sorry

end correct_height_l1391_139191


namespace not_always_preservable_flight_relations_l1391_139152

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents the flight guide for the country -/
structure FlightGuide where
  cities : Finset City
  has_direct_flight : City → City → Bool

/-- Represents a permutation of city IDs -/
def CityPermutation := Nat → Nat

/-- Theorem stating that it's not always possible to maintain flight relations after swapping city numbers -/
theorem not_always_preservable_flight_relations :
  ∃ (fg : FlightGuide) (m n : City),
    m ∈ fg.cities → n ∈ fg.cities → m ≠ n →
    ¬∀ (p : CityPermutation),
      (∀ c : City, c ∈ fg.cities → p (c.id) ≠ c.id → (c = m ∨ c = n)) →
      (p m.id = n.id ∧ p n.id = m.id) →
      (∀ c1 c2 : City, c1 ∈ fg.cities → c2 ∈ fg.cities →
        fg.has_direct_flight c1 c2 = fg.has_direct_flight
          ⟨p c1.id⟩ ⟨p c2.id⟩) :=
sorry

end not_always_preservable_flight_relations_l1391_139152


namespace movie_sale_price_is_10000_l1391_139110

/-- The sale price of a movie given costs and profit -/
def movie_sale_price (actor_cost food_cost_per_person equipment_cost_multiplier num_people profit : ℕ) : ℕ :=
  let food_cost := food_cost_per_person * num_people
  let equipment_cost := equipment_cost_multiplier * (actor_cost + food_cost)
  let total_cost := actor_cost + food_cost + equipment_cost
  total_cost + profit

/-- Theorem stating the sale price of the movie is $10000 -/
theorem movie_sale_price_is_10000 :
  movie_sale_price 1200 3 2 50 5950 = 10000 := by
  sorry

end movie_sale_price_is_10000_l1391_139110


namespace problem_solution_l1391_139129

def solution : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {(3, 2, 1, 3, 2, 1), (6, 1, 1, 2, 2, 2), (7, 1, 1, 3, 3, 1), (8, 1, 1, 5, 2, 1),
   (2, 2, 2, 6, 1, 1), (3, 3, 1, 7, 1, 1), (5, 2, 1, 8, 1, 1)}

def satisfies_conditions (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, x, y, z) := t
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

theorem problem_solution :
  ∀ t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ, satisfies_conditions t ↔ t ∈ solution := by
  sorry

#check problem_solution

end problem_solution_l1391_139129


namespace geometric_sequence_relation_l1391_139153

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c : ℝ)
  (isGeometric : ∃ r : ℝ, r ≠ 0 ∧ a = -2 * r ∧ b = a * r ∧ c = b * r ∧ -8 = c * r)

/-- The theorem stating the relationship between b and ac in the geometric sequence -/
theorem geometric_sequence_relation (seq : GeometricSequence) : seq.b = -4 ∧ seq.a * seq.c = 16 := by
  sorry

end geometric_sequence_relation_l1391_139153


namespace square_perimeter_l1391_139138

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 625) : 4 * s = 100 := by
  sorry

end square_perimeter_l1391_139138


namespace amount_in_paise_l1391_139167

theorem amount_in_paise : 
  let a : ℝ := 130
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := (percentage / 100) * a
  let paise_per_rupee : ℝ := 100
  (percentage / 100 * a) * paise_per_rupee = 65 := by
  sorry

end amount_in_paise_l1391_139167


namespace practice_time_proof_l1391_139155

/-- Calculates the required practice time for Friday given the practice times for Monday to Thursday and the total required practice time for the week. -/
def friday_practice_time (monday tuesday wednesday thursday total_time : ℕ) : ℕ :=
  total_time - (monday + tuesday + wednesday + thursday)

/-- Theorem stating that given the practice times for Monday to Thursday and the total required practice time, the remaining time for Friday is 60 minutes. -/
theorem practice_time_proof (total_time : ℕ) (h1 : total_time = 300) 
  (thursday : ℕ) (h2 : thursday = 50)
  (wednesday : ℕ) (h3 : wednesday = thursday + 5)
  (tuesday : ℕ) (h4 : tuesday = wednesday - 10)
  (monday : ℕ) (h5 : monday = 2 * tuesday) :
  friday_practice_time monday tuesday wednesday thursday total_time = 60 := by
  sorry

#eval friday_practice_time 90 45 55 50 300

end practice_time_proof_l1391_139155


namespace dinitrogen_pentoxide_molecular_weight_l1391_139195

/-- The molecular weight of Dinitrogen pentoxide in grams per mole. -/
def molecular_weight : ℝ := 108

/-- The number of moles given in the problem. -/
def given_moles : ℝ := 9

/-- The total weight of the given moles in grams. -/
def total_weight : ℝ := 972

/-- Theorem stating that the molecular weight of Dinitrogen pentoxide is 108 grams/mole. -/
theorem dinitrogen_pentoxide_molecular_weight :
  molecular_weight = total_weight / given_moles :=
sorry

end dinitrogen_pentoxide_molecular_weight_l1391_139195


namespace three_Z_five_l1391_139148

/-- The operation Z defined on real numbers -/
def Z (a b : ℝ) : ℝ := b + 7*a - 3*a^2

/-- Theorem stating that 3 Z 5 = -1 -/
theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end three_Z_five_l1391_139148


namespace solution_is_five_binomial_coefficient_identity_l1391_139114

-- Define A_x
def A (x : ℕ) : ℕ := x * (x - 1) * (x - 2)

-- Part 1: Prove that the solution to 3A_x^3 = 2A_{x+1}^2 + 6A_x^2 is x = 5
theorem solution_is_five : ∃ (x : ℕ), x > 3 ∧ 3 * (A x)^3 = 2 * (A (x + 1))^2 + 6 * (A x)^2 ∧ x = 5 := by
  sorry

-- Part 2: Prove that kC_n^k = nC_{n-1}^{k-1}
theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) : 
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end solution_is_five_binomial_coefficient_identity_l1391_139114


namespace square_sum_difference_equals_243_l1391_139108

theorem square_sum_difference_equals_243 : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 243 := by
  sorry

end square_sum_difference_equals_243_l1391_139108


namespace canteen_distance_l1391_139168

theorem canteen_distance (a b c x : ℝ) : 
  a = 400 → 
  c = 600 → 
  a^2 + b^2 = c^2 → 
  x^2 = a^2 + (b - x)^2 → 
  x = 410 := by
sorry

end canteen_distance_l1391_139168


namespace min_m_for_inequality_l1391_139127

theorem min_m_for_inequality : 
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m ≤ 1) ∧ 
  (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m' ≤ 1) → m' ≥ 3) :=
by sorry


end min_m_for_inequality_l1391_139127


namespace stoichiometric_ratio_l1391_139181

-- Define the reaction rates
variable (vA vB vC : ℝ)

-- Define the relationships between reaction rates
axiom rate_relation1 : vB = 3 * vA
axiom rate_relation2 : 3 * vC = 2 * vB

-- Define the stoichiometric coefficients
variable (a b c : ℕ)

-- Theorem: Given the rate relationships, prove the stoichiometric coefficient ratio
theorem stoichiometric_ratio : 
  vB = 3 * vA → 3 * vC = 2 * vB → a = 1 ∧ b = 3 ∧ c = 2 :=
by sorry

end stoichiometric_ratio_l1391_139181


namespace unique_solution_quadratic_l1391_139103

-- Define the quadratic equation
def quadratic_equation (x m : ℚ) : Prop :=
  3 * x^2 - 7 * x + m = 0

-- Define the condition for exactly one solution
def has_exactly_one_solution (m : ℚ) : Prop :=
  ∃! x, quadratic_equation x m

-- Theorem statement
theorem unique_solution_quadratic :
  ∀ m : ℚ, has_exactly_one_solution m → m = 49 / 12 :=
by sorry

end unique_solution_quadratic_l1391_139103


namespace fuel_cost_savings_l1391_139107

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) (journey_distance : ℝ)
  (h1 : efficiency_improvement = 0.6)
  (h2 : fuel_cost_increase = 0.25)
  (h3 : journey_distance = 1000) : 
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_journey_cost := journey_distance / old_efficiency * old_fuel_cost
  let new_journey_cost := journey_distance / new_efficiency * new_fuel_cost
  let percent_savings := (1 - new_journey_cost / old_journey_cost) * 100
  percent_savings = 21.875 := by
sorry

#eval (1 - (1000 / (1.6 * 1) * 1.25) / (1000 / 1 * 1)) * 100

end fuel_cost_savings_l1391_139107


namespace convex_ngon_division_constant_l1391_139119

/-- A convex n-gon can be divided into triangles using non-intersecting diagonals -/
structure ConvexNGonDivision (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (triangles : ℕ)
  (diagonals : ℕ)

/-- The number of triangles and diagonals in any division of a convex n-gon is constant -/
theorem convex_ngon_division_constant (n : ℕ) (d : ConvexNGonDivision n) :
  d.triangles = n - 2 ∧ d.diagonals = n - 3 :=
sorry

end convex_ngon_division_constant_l1391_139119


namespace min_value_theorem_l1391_139102

theorem min_value_theorem (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ (∃ y, 8 * y^2 + 7 * y + 6 = 5 ∧ 3 * y + 2 = m) ∧ m = -1 := by
  sorry

end min_value_theorem_l1391_139102


namespace binomial_coefficient_17_8_l1391_139188

theorem binomial_coefficient_17_8 :
  (Nat.choose 15 6 = 5005) →
  (Nat.choose 15 7 = 6435) →
  (Nat.choose 15 8 = 6435) →
  Nat.choose 17 8 = 24310 := by
  sorry

end binomial_coefficient_17_8_l1391_139188


namespace katie_marbles_count_l1391_139180

def pink_marbles : ℕ := 13

def orange_marbles (pink : ℕ) : ℕ := pink - 9

def purple_marbles (orange : ℕ) : ℕ := 4 * orange

def total_marbles (pink orange purple : ℕ) : ℕ := pink + orange + purple

theorem katie_marbles_count :
  total_marbles pink_marbles (orange_marbles pink_marbles) (purple_marbles (orange_marbles pink_marbles)) = 33 :=
by
  sorry


end katie_marbles_count_l1391_139180


namespace mark_change_factor_l1391_139100

theorem mark_change_factor (n : ℕ) (initial_avg final_avg : ℚ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : final_avg = 140) :
  (n * final_avg) / (n * initial_avg) = 2 :=
sorry

end mark_change_factor_l1391_139100


namespace negation_equivalence_l1391_139178

theorem negation_equivalence (a b : ℝ) : 
  ¬(a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0) := by
  sorry

end negation_equivalence_l1391_139178


namespace age_ratio_problem_l1391_139172

/-- Given two people A and B, where:
    1. The ratio of their present ages is 6:3
    2. The ratio between A's age at a certain point in the past and B's age at a certain point in the future is the same as their present ratio
    3. The ratio between A's age 4 years hence and B's age 4 years ago is 5
    Prove that the ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
theorem age_ratio_problem (a b : ℕ) (h1 : a = 2 * b) 
  (h2 : ∀ (x y : ℤ), a + x = 2 * (b + y))
  (h3 : (a + 4) / (b - 4 : ℚ) = 5) :
  (a - 4 : ℚ) / (b + 4) = 1 := by
  sorry

end age_ratio_problem_l1391_139172


namespace sam_eating_period_l1391_139149

def apples_per_sandwich : ℕ := 4
def sandwiches_per_day : ℕ := 10
def total_apples : ℕ := 280

theorem sam_eating_period :
  (total_apples / (apples_per_sandwich * sandwiches_per_day) : ℕ) = 7 :=
sorry

end sam_eating_period_l1391_139149


namespace bowling_team_weight_l1391_139137

theorem bowling_team_weight (x : ℝ) : 
  let initial_players : ℕ := 7
  let initial_avg_weight : ℝ := 94
  let new_players : ℕ := 2
  let known_new_player_weight : ℝ := 60
  let new_avg_weight : ℝ := 92
  (initial_players * initial_avg_weight + x + known_new_player_weight) / 
    (initial_players + new_players) = new_avg_weight → x = 110 :=
by sorry

end bowling_team_weight_l1391_139137


namespace right_prism_cross_section_type_l1391_139113

/-- Represents a right prism -/
structure RightPrism where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a cross-section of a prism -/
inductive CrossSection
  | GeneralTrapezoid
  | IsoscelesTrapezoid
  | Other

/-- Function to determine the type of cross-section through the centers of base faces -/
def crossSectionThroughCenters (prism : RightPrism) : CrossSection :=
  sorry

/-- Theorem stating that the cross-section through the centers of base faces
    of a right prism is either a general trapezoid or an isosceles trapezoid -/
theorem right_prism_cross_section_type (prism : RightPrism) :
  (crossSectionThroughCenters prism = CrossSection.GeneralTrapezoid) ∨
  (crossSectionThroughCenters prism = CrossSection.IsoscelesTrapezoid) :=
by
  sorry

end right_prism_cross_section_type_l1391_139113


namespace triangle_area_unchanged_l1391_139185

theorem triangle_area_unchanged 
  (base height : ℝ) 
  (base_positive : base > 0) 
  (height_positive : height > 0) : 
  (1/2) * base * height = (1/2) * (base / 3) * (3 * height) := by
  sorry

end triangle_area_unchanged_l1391_139185


namespace at_least_ten_mutual_reports_l1391_139111

-- Define the type for spies
def Spy : Type := ℕ

-- Define the total number of spies
def total_spies : ℕ := 20

-- Define the number of colleagues each spy reports on
def reports_per_spy : ℕ := 10

-- Define the reporting relation
def reports_on (s₁ s₂ : Spy) : Prop := sorry

-- State the theorem
theorem at_least_ten_mutual_reports :
  ∃ (mutual_reports : Finset (Spy × Spy)),
    (∀ (pair : Spy × Spy), pair ∈ mutual_reports →
      reports_on pair.1 pair.2 ∧ reports_on pair.2 pair.1) ∧
    mutual_reports.card ≥ 10 := by
  sorry

end at_least_ten_mutual_reports_l1391_139111


namespace solution_to_equation_l1391_139101

theorem solution_to_equation : ∃ x : ℝ, ((18 + x) / 3 + 10) / 5 = 4 ∧ x = 12 := by
  sorry

end solution_to_equation_l1391_139101


namespace fixed_point_theorem_l1391_139182

/-- A line passes through a point if the point's coordinates satisfy the line equation -/
def PassesThrough (m : ℝ) (x y : ℝ) : Prop := m * x - y + 3 = 0

/-- The theorem states that for all real numbers m, 
    the line mx - y + 3 = 0 passes through the point (0, 3) -/
theorem fixed_point_theorem : ∀ m : ℝ, PassesThrough m 0 3 := by
  sorry

end fixed_point_theorem_l1391_139182


namespace polynomial_division_theorem_l1391_139136

theorem polynomial_division_theorem (a b c : ℚ) : 
  (∀ x, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end polynomial_division_theorem_l1391_139136


namespace line_inclination_angle_l1391_139151

/-- The inclination angle of a line given by parametric equations -/
def inclinationAngle (x y : ℝ → ℝ) : ℝ := sorry

/-- Cosine of 20 degrees -/
def cos20 : ℝ := sorry

/-- Sine of 20 degrees -/
def sin20 : ℝ := sorry

theorem line_inclination_angle :
  let x : ℝ → ℝ := λ t => -t * cos20
  let y : ℝ → ℝ := λ t => 3 + t * sin20
  inclinationAngle x y = 160 * π / 180 := by sorry

end line_inclination_angle_l1391_139151


namespace factorial_6_eq_720_l1391_139160

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_eq_720 : factorial 6 = 720 := by
  sorry

end factorial_6_eq_720_l1391_139160


namespace rectangle_area_l1391_139130

/-- The area of a rectangle with width 5.4 meters and height 2.5 meters is 13.5 square meters. -/
theorem rectangle_area : 
  let width : Real := 5.4
  let height : Real := 2.5
  width * height = 13.5 := by
  sorry

end rectangle_area_l1391_139130


namespace smallest_integer_proof_l1391_139165

def club_size : ℕ := 30

def smallest_integer : ℕ := 2329089562800

theorem smallest_integer_proof :
  (∀ i ∈ Finset.range 28, smallest_integer % i = 0) ∧
  (smallest_integer % 31 = 0) ∧
  (∀ i ∈ Finset.range 3, smallest_integer % (28 + i) ≠ 0) ∧
  (∀ n : ℕ, n < smallest_integer →
    ¬((∀ i ∈ Finset.range 28, n % i = 0) ∧
      (n % 31 = 0) ∧
      (∀ i ∈ Finset.range 3, n % (28 + i) ≠ 0))) :=
by sorry

#check smallest_integer_proof

end smallest_integer_proof_l1391_139165


namespace cheryl_expense_difference_l1391_139157

def electricity_bill : ℝ := 800
def golf_tournament_payment : ℝ := 1440

def monthly_cell_phone_expenses (x : ℝ) : ℝ := electricity_bill + x

def golf_tournament_cost (x : ℝ) : ℝ := 1.2 * monthly_cell_phone_expenses x

theorem cheryl_expense_difference :
  ∃ x : ℝ, 
    x = 400 ∧ 
    golf_tournament_cost x = golf_tournament_payment :=
sorry

end cheryl_expense_difference_l1391_139157


namespace paving_cost_calculation_l1391_139140

/-- Calculates the cost of paving a rectangular floor -/
def calculate_paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m x 4m room at 850 Rs/m² is 18700 Rs -/
theorem paving_cost_calculation :
  calculate_paving_cost 5.5 4 850 = 18700 := by
  sorry

end paving_cost_calculation_l1391_139140


namespace expression_evaluation_l1391_139105

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((2*x + y)^2 - y*(y + 4*x) - 8*x) / (-2*x) = 8 := by
  sorry

end expression_evaluation_l1391_139105


namespace largest_number_l1391_139183

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def number_A : Nat := to_base_10 [8, 5] 9
def number_B : Nat := to_base_10 [2, 0, 0] 6
def number_C : Nat := to_base_10 [6, 8] 8
def number_D : Nat := 70

theorem largest_number :
  number_A > number_B ∧ number_A > number_C ∧ number_A > number_D := by
  sorry

end largest_number_l1391_139183


namespace imaginary_part_of_z_l1391_139131

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I) / Complex.I) = -1 := by
  sorry

end imaginary_part_of_z_l1391_139131


namespace kite_area_16_20_l1391_139156

/-- Calculates the area of a kite given its base and height -/
def kite_area (base : ℝ) (height : ℝ) : ℝ :=
  base * height

/-- Theorem: The area of a kite with base 16 inches and height 20 inches is 160 square inches -/
theorem kite_area_16_20 :
  kite_area 16 20 = 160 := by
sorry

end kite_area_16_20_l1391_139156


namespace conditionA_not_necessary_nor_sufficient_l1391_139123

/-- Condition A: The square root of 1 plus sine of theta equals a -/
def conditionA (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: The sine of half theta plus the cosine of half theta equals a -/
def conditionB (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that Condition A is neither necessary nor sufficient for Condition B -/
theorem conditionA_not_necessary_nor_sufficient :
  ¬(∀ θ a, conditionB θ a → conditionA θ a) ∧
  ¬(∀ θ a, conditionA θ a → conditionB θ a) :=
sorry

end conditionA_not_necessary_nor_sufficient_l1391_139123


namespace movie_theater_tickets_l1391_139104

theorem movie_theater_tickets (adult_price child_price total_revenue adult_tickets : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_revenue = 5100)
  (h4 : adult_tickets = 500) :
  ∃ child_tickets : ℕ, 
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    adult_tickets + child_tickets = 900 :=
by sorry

end movie_theater_tickets_l1391_139104


namespace square_of_negative_product_l1391_139173

theorem square_of_negative_product (a b : ℝ) : (-a * b)^2 = a^2 * b^2 := by
  sorry

end square_of_negative_product_l1391_139173


namespace least_x_for_divisibility_l1391_139141

theorem least_x_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) → x = 2 := by
  sorry

end least_x_for_divisibility_l1391_139141


namespace system_of_inequalities_l1391_139112

theorem system_of_inequalities (x : ℝ) : 2*x + 1 > x ∧ x < -3*x + 8 → -1 < x ∧ x < 2 := by
  sorry

end system_of_inequalities_l1391_139112


namespace not_p_and_q_implies_at_most_one_true_l1391_139163

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end not_p_and_q_implies_at_most_one_true_l1391_139163


namespace max_area_rectangle_with_perimeter_52_l1391_139125

/-- The maximum area of a rectangle with a perimeter of 52 centimeters is 169 square centimeters. -/
theorem max_area_rectangle_with_perimeter_52 :
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 52 →
  length * width ≤ 169 := by
sorry

end max_area_rectangle_with_perimeter_52_l1391_139125


namespace sum_in_base5_l1391_139109

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem sum_in_base5 :
  toBase5 (toDecimal [2, 1, 3] + toDecimal [3, 2, 4] + toDecimal [1, 4, 1]) = [1, 3, 3, 3] :=
sorry

end sum_in_base5_l1391_139109


namespace f_simplification_and_range_l1391_139198

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 2 * Real.sin x - 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 2)

theorem f_simplification_and_range : 
  ∀ x : ℝ, Real.sin x ≠ 2 → 
    (f x = Real.sin x ^ 2 + 4 * Real.sin x + 6) ∧ 
    (∃ y : ℝ, f y = 1) ∧ 
    (∃ z : ℝ, f z = 13) ∧ 
    (∀ w : ℝ, Real.sin w ≠ 2 → 1 ≤ f w ∧ f w ≤ 13) :=
by sorry

end f_simplification_and_range_l1391_139198


namespace suzanne_reading_difference_l1391_139132

/-- Represents the number of pages Suzanne read on Tuesday -/
def pages_tuesday (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  total_pages - monday_pages - remaining_pages

/-- The difference in pages read between Tuesday and Monday -/
def pages_difference (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  pages_tuesday total_pages monday_pages remaining_pages - monday_pages

theorem suzanne_reading_difference :
  pages_difference 64 15 18 = 16 := by sorry

end suzanne_reading_difference_l1391_139132


namespace cube_with_72cm_edges_l1391_139171

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of a cube -/
def Cube.sumOfEdges (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

/-- Theorem stating the properties of a cube with sum of edges 72 cm -/
theorem cube_with_72cm_edges (c : Cube) 
  (h : c.sumOfEdges = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end cube_with_72cm_edges_l1391_139171


namespace annabelle_allowance_l1391_139174

/-- Proves that Annabelle's weekly allowance is $30 given the problem conditions -/
theorem annabelle_allowance :
  ∀ A : ℚ, (1/3 : ℚ) * A + 8 + 12 = A → A = 30 := by
  sorry

end annabelle_allowance_l1391_139174


namespace truck_driver_gas_cost_l1391_139184

/-- A truck driver's gas cost problem -/
theorem truck_driver_gas_cost 
  (miles_per_gallon : ℝ) 
  (miles_per_hour : ℝ) 
  (pay_per_mile : ℝ) 
  (total_pay : ℝ) 
  (drive_time : ℝ) 
  (h1 : miles_per_gallon = 10)
  (h2 : miles_per_hour = 30)
  (h3 : pay_per_mile = 0.5)
  (h4 : total_pay = 90)
  (h5 : drive_time = 10) :
  (total_pay / (miles_per_hour * drive_time / miles_per_gallon)) = 3 := by
sorry


end truck_driver_gas_cost_l1391_139184


namespace abs_neg_product_eq_product_l1391_139197

theorem abs_neg_product_eq_product {a b : ℝ} (ha : a < 0) (hb : 0 < b) : |-(a * b)| = a * b := by
  sorry

end abs_neg_product_eq_product_l1391_139197


namespace cricket_team_left_handed_fraction_l1391_139164

theorem cricket_team_left_handed_fraction :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 55 →
    throwers = 37 →
    right_handed = 49 →
    (total_players - throwers : ℚ) ≠ 0 →
    (left_handed_non_throwers : ℚ) / (total_players - throwers) = 1 / 3 :=
  λ total_players throwers right_handed
    h_total h_throwers h_right_handed h_non_zero ↦ by
  sorry

end cricket_team_left_handed_fraction_l1391_139164


namespace equation_solution_l1391_139122

theorem equation_solution : 
  ∃ x : ℚ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) ∧ (x = -80 / 19) := by
  sorry

end equation_solution_l1391_139122


namespace aarti_work_completion_time_l1391_139117

/-- If Aarti can complete three times a piece of work in 24 days, 
    then she can complete one piece of work in 8 days. -/
theorem aarti_work_completion_time : 
  ∀ (work_time : ℝ), work_time > 0 → 3 * work_time = 24 → work_time = 8 := by
  sorry

end aarti_work_completion_time_l1391_139117


namespace cone_symmetry_properties_l1391_139142

-- Define the types of cones
inductive ConeType
  | Bounded
  | UnboundedSingleNapped
  | UnboundedDoubleNapped

-- Define symmetry properties
structure SymmetryProperties where
  hasAxis : Bool
  hasPlaneBundleThroughAxis : Bool
  hasCentralSymmetry : Bool
  hasPerpendicularPlane : Bool

-- Function to determine symmetry properties based on cone type
def symmetryPropertiesForCone (coneType : ConeType) : SymmetryProperties :=
  match coneType with
  | ConeType.Bounded => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedSingleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedDoubleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := true,
      hasPerpendicularPlane := true
    }

theorem cone_symmetry_properties (coneType : ConeType) :
  (coneType = ConeType.Bounded ∨ coneType = ConeType.UnboundedSingleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = false ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = false
  ∧
  (coneType = ConeType.UnboundedDoubleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = true ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = true :=
by sorry

end cone_symmetry_properties_l1391_139142


namespace parabola_properties_l1391_139176

/-- Represents a parabola of the form y = ax^2 + 4ax + 3 -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- The x-coordinate of the axis of symmetry of the parabola -/
def Parabola.axisOfSymmetry (p : Parabola) : ℝ := -2

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + 4 * p.a * x + 3

theorem parabola_properties (p : Parabola) :
  (p.axisOfSymmetry = -2) ∧
  p.isOnParabola 0 3 := by sorry

end parabola_properties_l1391_139176


namespace aluminium_hydroxide_weight_l1391_139147

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of moles of Aluminium hydroxide -/
def num_moles : ℝ := 4

/-- The molecular weight of Aluminium hydroxide (Al(OH)₃) in g/mol -/
def molecular_weight_AlOH3 : ℝ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- The total weight of the given number of moles of Aluminium hydroxide in grams -/
def total_weight : ℝ := num_moles * molecular_weight_AlOH3

theorem aluminium_hydroxide_weight :
  total_weight = 312.04 := by sorry

end aluminium_hydroxide_weight_l1391_139147


namespace ben_lighter_than_carl_l1391_139133

/-- Given the weights of several people and their relationships, prove that Ben is 16 pounds lighter than Carl. -/
theorem ben_lighter_than_carl (al ben carl ed : ℕ) : 
  al = ben + 25 →  -- Al is 25 pounds heavier than Ben
  ed = 146 →       -- Ed weighs 146 pounds
  al = ed + 38 →   -- Ed is 38 pounds lighter than Al
  carl = 175 →     -- Carl weighs 175 pounds
  carl - ben = 16  -- Ben is 16 pounds lighter than Carl
:= by sorry

end ben_lighter_than_carl_l1391_139133


namespace q_factor_change_l1391_139154

theorem q_factor_change (e x z : ℝ) (h : x ≠ 0 ∧ z ≠ 0) :
  let q := 5 * e / (4 * x * z^2)
  let q_new := 5 * (4 * e) / (4 * (2 * x) * (3 * z)^2)
  q_new = (4 / 9) * q :=
by
  sorry

end q_factor_change_l1391_139154


namespace households_using_only_brand_A_l1391_139139

/-- The number of households that use only brand A soap -/
def only_brand_A : ℕ := 60

/-- The number of households that use only brand B soap -/
def only_brand_B : ℕ := 75

/-- The number of households that use both brand A and brand B soap -/
def both_brands : ℕ := 25

/-- The number of households that use neither brand A nor brand B soap -/
def neither_brand : ℕ := 80

/-- The total number of households surveyed -/
def total_households : ℕ := 240

/-- Theorem stating that the number of households using only brand A soap is 60 -/
theorem households_using_only_brand_A :
  only_brand_A = total_households - only_brand_B - both_brands - neither_brand :=
by sorry

end households_using_only_brand_A_l1391_139139


namespace function_passes_through_point_l1391_139193

theorem function_passes_through_point (a : ℝ) (h : a < 0) :
  let f := fun x => (1 - a)^x - 1
  f 0 = -1 := by sorry

end function_passes_through_point_l1391_139193


namespace gcd_lcm_product_24_40_l1391_139121

theorem gcd_lcm_product_24_40 : Nat.gcd 24 40 * Nat.lcm 24 40 = 960 := by
  sorry

end gcd_lcm_product_24_40_l1391_139121


namespace disjoint_subsets_remainder_l1391_139159

def T : Finset Nat := Finset.range 15

def m : Nat :=
  (3^15 - 2 * 2^15 + 1) / 2

theorem disjoint_subsets_remainder : m % 1000 = 686 := by
  sorry

end disjoint_subsets_remainder_l1391_139159


namespace grape_juice_mixture_l1391_139146

theorem grape_juice_mixture (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 50 →
  initial_percentage = 0.1 →
  added_volume = 10 →
  let initial_grape_juice := initial_volume * initial_percentage
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  let final_percentage := total_grape_juice / final_volume
  final_percentage = 0.25 := by sorry

end grape_juice_mixture_l1391_139146


namespace rectangle_to_circle_area_l1391_139177

/-- Given a rectangle with area 200 square units and one side 5 units longer than twice the other side,
    the area of the largest circle that can be formed from a string equal in length to the rectangle's perimeter
    is 400/π square units. -/
theorem rectangle_to_circle_area (x : ℝ) (h1 : x > 0) (h2 : x * (2 * x + 5) = 200) : 
  let perimeter := 2 * (x + (2 * x + 5))
  (perimeter / (2 * Real.pi))^2 * Real.pi = 400 / Real.pi :=
by sorry

end rectangle_to_circle_area_l1391_139177


namespace amaya_total_marks_l1391_139170

def total_marks (music maths arts social_studies : ℕ) : ℕ :=
  music + maths + arts + social_studies

theorem amaya_total_marks :
  ∀ (music maths arts social_studies : ℕ),
    music = 70 →
    maths = music - music / 10 →
    arts = maths + 20 →
    social_studies = music + 10 →
    total_marks music maths arts social_studies = 296 :=
by
  sorry

end amaya_total_marks_l1391_139170


namespace system_of_inequalities_l1391_139166

theorem system_of_inequalities (x : ℝ) : 
  (x + 1 < 5 ∧ (2 * x - 1) / 3 ≥ 1) ↔ 2 ≤ x ∧ x < 4 := by
  sorry

end system_of_inequalities_l1391_139166


namespace circle_on_parabola_fixed_point_l1391_139175

/-- A circle with center on a parabola and tangent to a line passes through a fixed point -/
theorem circle_on_parabola_fixed_point (h k : ℝ) :
  k = (1/12) * h^2 →  -- Center (h, k) lies on the parabola y = (1/12)x^2
  (k + 3)^2 = h^2 + (k - 3)^2 →  -- Circle is tangent to the line y + 3 = 0
  (0 - h)^2 + (3 - k)^2 = (k + 3)^2 :=  -- Point (0, 3) lies on the circle
by sorry

end circle_on_parabola_fixed_point_l1391_139175


namespace walnut_trees_after_planting_l1391_139116

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees planted_trees : ℕ) : ℕ :=
  initial_trees + planted_trees

/-- Theorem: The total number of walnut trees after planting is 55 -/
theorem walnut_trees_after_planting :
  total_trees 22 33 = 55 := by
  sorry

end walnut_trees_after_planting_l1391_139116


namespace smallest_angle_of_dividable_isosceles_triangle_l1391_139179

-- Define an isosceles triangle
structure IsoscelesTriangle where
  α : ℝ
  -- The base angles are equal (α) and the sum of all angles is 180°
  angleSum : α + α + (180 - 2*α) = 180

-- Define a function that checks if a triangle can be divided into two isosceles triangles
def canDivideIntoTwoIsosceles (t : IsoscelesTriangle) : Prop :=
  -- This is a placeholder for the actual condition
  -- In reality, this would involve a more complex geometric condition
  true

-- Theorem statement
theorem smallest_angle_of_dividable_isosceles_triangle :
  ∀ t : IsoscelesTriangle, 
    canDivideIntoTwoIsosceles t → 
    (min t.α (180 - 2*t.α) ≥ 180 / 7) ∧ 
    (∃ t' : IsoscelesTriangle, canDivideIntoTwoIsosceles t' ∧ min t'.α (180 - 2*t'.α) = 180 / 7) :=
sorry

end smallest_angle_of_dividable_isosceles_triangle_l1391_139179


namespace candidates_per_state_l1391_139124

theorem candidates_per_state : 
  ∀ (x : ℕ), 
    (x * 6 / 100 : ℚ) + 80 = (x * 7 / 100 : ℚ) → 
    x = 8000 := by
  sorry

end candidates_per_state_l1391_139124


namespace competition_finish_orders_l1391_139144

theorem competition_finish_orders (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end competition_finish_orders_l1391_139144


namespace n_value_l1391_139196

theorem n_value (e n : ℕ+) : 
  (Nat.lcm e n = 690) →
  (100 ≤ n) →
  (n < 1000) →
  (¬ 3 ∣ n) →
  (¬ 2 ∣ e) →
  (n = 230) :=
sorry

end n_value_l1391_139196
