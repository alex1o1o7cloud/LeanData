import Mathlib

namespace set_classification_l2991_299156

/-- The set of numbers we're working with -/
def S : Set ℝ := {-2, -3.14, 0.3, 0, Real.pi/3, 22/7, -0.1212212221}

/-- The set of positive numbers in S -/
def positiveS : Set ℝ := {x ∈ S | x > 0}

/-- The set of negative numbers in S -/
def negativeS : Set ℝ := {x ∈ S | x < 0}

/-- The set of integers in S -/
def integerS : Set ℝ := {x ∈ S | ∃ n : ℤ, x = n}

/-- The set of rational numbers in S -/
def rationalS : Set ℝ := {x ∈ S | ∃ p q : ℤ, q ≠ 0 ∧ x = p / q}

theorem set_classification :
  positiveS = {0.3, Real.pi/3, 22/7} ∧
  negativeS = {-2, -3.14, -0.1212212221} ∧
  integerS = {-2, 0} ∧
  rationalS = {-2, 0, 0.3, 22/7} := by
  sorry

end set_classification_l2991_299156


namespace red_light_estimation_l2991_299159

theorem red_light_estimation (total_students : ℕ) (total_yes : ℕ) (known_yes_rate : ℚ) :
  total_students = 600 →
  total_yes = 180 →
  known_yes_rate = 1/2 →
  ∃ (estimated_red_light : ℕ), estimated_red_light = 60 :=
by sorry

end red_light_estimation_l2991_299159


namespace smallest_base_for_145_l2991_299104

theorem smallest_base_for_145 :
  ∀ b : ℕ, b ≥ 2 →
    (∀ n : ℕ, n ≥ 2 ∧ n < b → n^2 ≤ 145 ∧ 145 < n^3) →
    b = 13 := by
  sorry

end smallest_base_for_145_l2991_299104


namespace a_greater_than_b_l2991_299142

theorem a_greater_than_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq1 : a^3 = a + 1) (eq2 : b^6 = b + 3*a) : a > b := by
  sorry

end a_greater_than_b_l2991_299142


namespace farthest_poles_distance_l2991_299158

/-- The number of utility poles -/
def num_poles : ℕ := 45

/-- The interval between each pole in meters -/
def interval : ℕ := 60

/-- The distance between the first and last pole in kilometers -/
def distance : ℚ := 2.64

theorem farthest_poles_distance :
  (((num_poles - 1) * interval) : ℚ) / 1000 = distance := by sorry

end farthest_poles_distance_l2991_299158


namespace murtha_pebble_collection_l2991_299130

def pebbles_on_day (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * (n - 1) + 2

def total_pebbles (days : ℕ) : ℕ :=
  (List.range days).map pebbles_on_day |>.sum

theorem murtha_pebble_collection :
  total_pebbles 15 = 345 := by
  sorry

end murtha_pebble_collection_l2991_299130


namespace hemisphere_surface_area_l2991_299121

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  let sphere_area := 4 * π * r^2
  let base_area := π * r^2
  let excluded_base_area := (1/4) * base_area
  let hemisphere_curved_area := (1/2) * sphere_area
  hemisphere_curved_area + base_area - excluded_base_area = 275 * π := by
  sorry

end hemisphere_surface_area_l2991_299121


namespace sum_of_a_and_b_is_three_l2991_299179

theorem sum_of_a_and_b_is_three (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b - i * a) : a + b = 3 := by
  sorry

end sum_of_a_and_b_is_three_l2991_299179


namespace basketball_only_count_l2991_299166

theorem basketball_only_count (total : ℕ) (basketball : ℕ) (table_tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_basketball : basketball = 15)
  (h_table_tennis : table_tennis = 10)
  (h_neither : neither = 8)
  (h_sum : total = basketball + table_tennis - (basketball + table_tennis - total + neither) + neither) :
  basketball - (basketball + table_tennis - total + neither) = 12 := by
  sorry

end basketball_only_count_l2991_299166


namespace triangle_theorem_l2991_299194

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * Real.sin t.C - Real.sin t.B) / Real.sin t.B = (t.a * Real.cos t.B) / (t.b * Real.cos t.A))
  (h2 : t.a = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.B) :
  t.A = π/3 ∧ t.b = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end triangle_theorem_l2991_299194


namespace one_third_1206_percent_of_200_l2991_299167

theorem one_third_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end one_third_1206_percent_of_200_l2991_299167


namespace odd_symmetric_latin_square_diagonal_l2991_299172

/-- A square matrix of size n × n filled with integers from 1 to n -/
def LatinSquare (n : ℕ) := Matrix (Fin n) (Fin n) (Fin n)

/-- Predicate to check if a LatinSquare has all numbers from 1 to n in each row and column -/
def is_valid_latin_square (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, (∃ k : Fin n, A i k = j) ∧ (∃ k : Fin n, A k j = i)

/-- Predicate to check if a LatinSquare is symmetric -/
def is_symmetric (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, A i j = A j i

/-- Predicate to check if all numbers from 1 to n appear on the main diagonal -/
def all_on_diagonal (A : LatinSquare n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, A i i = k

/-- Theorem stating that for odd n, a valid symmetric Latin square has all numbers on its diagonal -/
theorem odd_symmetric_latin_square_diagonal (n : ℕ) (hn : Odd n) (A : LatinSquare n)
  (hvalid : is_valid_latin_square A) (hsym : is_symmetric A) :
  all_on_diagonal A :=
sorry

end odd_symmetric_latin_square_diagonal_l2991_299172


namespace journey_fraction_l2991_299154

theorem journey_fraction (total_journey : ℝ) (bus_fraction : ℝ) (foot_distance : ℝ)
  (h1 : total_journey = 130)
  (h2 : bus_fraction = 17 / 20)
  (h3 : foot_distance = 6.5) :
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1 / 10 :=
by
  sorry

end journey_fraction_l2991_299154


namespace initial_profit_percentage_is_correct_l2991_299163

/-- Represents the profit percentage as a real number between 0 and 1 -/
def ProfitPercentage : Type := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The cost price of the book -/
def costPrice : ℝ := 300

/-- The additional amount added to the initial selling price -/
def additionalAmount : ℝ := 18

/-- The profit percentage if the book is sold with the additional amount -/
def newProfitPercentage : ProfitPercentage := ⟨0.18, by sorry⟩

/-- Calculate the selling price given a profit percentage -/
def sellingPrice (p : ProfitPercentage) : ℝ :=
  costPrice * (1 + p.val)

/-- The initial profit percentage -/
def initialProfitPercentage : ProfitPercentage := ⟨0.12, by sorry⟩

/-- Theorem stating that the initial profit percentage is correct -/
theorem initial_profit_percentage_is_correct :
  sellingPrice initialProfitPercentage + additionalAmount =
  sellingPrice newProfitPercentage :=
by sorry

end initial_profit_percentage_is_correct_l2991_299163


namespace intersection_parallel_perpendicular_line_l2991_299147

/-- The equation of a line passing through the intersection of two lines,
    parallel to one line, and perpendicular to another line. -/
theorem intersection_parallel_perpendicular_line 
  (l1 l2 l_parallel l_perpendicular : ℝ → ℝ → Prop) 
  (h_l1 : ∀ x y, l1 x y ↔ 2*x - 3*y + 10 = 0)
  (h_l2 : ∀ x y, l2 x y ↔ 3*x + 4*y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ x - y + 1 = 0)
  (h_perpendicular : ∀ x y, l_perpendicular x y ↔ 3*x - y - 2 = 0)
  : ∃ l : ℝ → ℝ → Prop, 
    (∃ x y, l1 x y ∧ l2 x y ∧ l x y) ∧ 
    (∀ x y, l x y ↔ x - y + 4 = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (1) + (-1) * (d - b) = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (3) + (-1) * (d - b) = 0) :=
by sorry

end intersection_parallel_perpendicular_line_l2991_299147


namespace fair_haired_women_percentage_l2991_299143

/-- Given that 32% of employees are women with fair hair and 80% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage 
  (total_employees : ℝ) 
  (h1 : total_employees > 0)
  (women_fair_hair : ℝ) 
  (h2 : women_fair_hair = 0.32 * total_employees)
  (fair_haired : ℝ) 
  (h3 : fair_haired = 0.80 * total_employees) :
  women_fair_hair / fair_haired = 0.40 := by
sorry

end fair_haired_women_percentage_l2991_299143


namespace rhombus_perimeter_l2991_299160

/-- The perimeter of a rhombus with diagonals of 18 inches and 32 inches is 4√337 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 32) :
  4 * (((d1 / 2) ^ 2 + (d2 / 2) ^ 2).sqrt) = 4 * Real.sqrt 337 := by
  sorry

end rhombus_perimeter_l2991_299160


namespace weights_representation_l2991_299153

def weights : List ℤ := [1, 3, 9, 27]

def is_representable (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧
    n = 27*a + 9*b + 3*c + d

theorem weights_representation :
  ∀ n : ℤ, 0 ≤ n → n < 41 → is_representable n :=
by sorry

end weights_representation_l2991_299153


namespace james_older_brother_age_is_16_l2991_299139

/-- The age of James' older brother given the conditions in the problem -/
def james_older_brother_age (john_current_age : ℕ) : ℕ :=
  let john_age_3_years_ago := john_current_age - 3
  let james_age_in_6_years := john_age_3_years_ago / 2
  let james_current_age := james_age_in_6_years - 6
  james_current_age + 4

/-- Theorem stating that James' older brother's age is 16 -/
theorem james_older_brother_age_is_16 :
  james_older_brother_age 39 = 16 := by
  sorry

end james_older_brother_age_is_16_l2991_299139


namespace complex_equation_solution_l2991_299181

theorem complex_equation_solution :
  ∃ z : ℂ, 4 + 2 * Complex.I * z = 3 - 5 * Complex.I * z ∧ z = Complex.I / 7 := by
  sorry

end complex_equation_solution_l2991_299181


namespace range_of_a_when_A_union_B_equals_A_l2991_299145

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- State the theorem
theorem range_of_a_when_A_union_B_equals_A :
  ∀ a : ℝ, (A ∪ B a = A) → a ≥ (1/2 : ℝ) := by sorry

end range_of_a_when_A_union_B_equals_A_l2991_299145


namespace doctors_visit_cost_is_250_l2991_299198

/-- Calculates the cost of a doctor's visit given the following conditions:
  * Number of vaccines needed
  * Cost per vaccine
  * Insurance coverage percentage
  * Cost of the trip
  * Total amount paid by Tom
-/
def doctors_visit_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (insurance_coverage : ℚ) 
                       (trip_cost : ℚ) (total_paid : ℚ) : ℚ :=
  let total_vaccine_cost := num_vaccines * vaccine_cost
  let medical_bills := total_vaccine_cost + (total_paid - trip_cost) / (1 - insurance_coverage)
  medical_bills - total_vaccine_cost

/-- Proves that the cost of the doctor's visit is $250 given the specified conditions -/
theorem doctors_visit_cost_is_250 : 
  doctors_visit_cost 10 45 0.8 1200 1340 = 250 := by
  sorry

end doctors_visit_cost_is_250_l2991_299198


namespace odd_periodic_sum_zero_l2991_299183

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_sum_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_periodic : is_periodic f 2) :
  f 1 + f 4 + f 7 = 0 := by
  sorry

end odd_periodic_sum_zero_l2991_299183


namespace james_final_amount_proof_l2991_299152

/-- The amount of money owned by James after paying off Lucas' debt -/
def james_final_amount : ℝ := 170

/-- The total amount owned by Lucas, James, and Ali -/
def total_amount : ℝ := 300

/-- The amount of Lucas' debt -/
def lucas_debt : ℝ := 25

/-- The difference between James' and Ali's initial amounts -/
def james_ali_difference : ℝ := 40

theorem james_final_amount_proof :
  ∃ (ali james lucas : ℝ),
    ali + james + lucas = total_amount ∧
    james = ali + james_ali_difference ∧
    lucas = -lucas_debt ∧
    james - (lucas_debt / 2) = james_final_amount :=
sorry

end james_final_amount_proof_l2991_299152


namespace thirteenth_term_l2991_299155

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 1 + a 9 = 16) ∧
  (a 4 = 1)

/-- The 13th term of the arithmetic sequence is 64 -/
theorem thirteenth_term (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 13 = 64 := by
  sorry

end thirteenth_term_l2991_299155


namespace simplify_expression_l2991_299128

theorem simplify_expression (a b c : ℝ) (h1 : 1 - a * b ≠ 0) (h2 : 1 + c * a ≠ 0) :
  ((a + b) / (1 - a * b) + (c - a) / (1 + c * a)) / (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) =
  (b + c) / (1 - b * c) :=
by sorry

end simplify_expression_l2991_299128


namespace sandbox_sand_calculation_l2991_299124

/-- Calculates the amount of sand needed to fill a square sandbox -/
theorem sandbox_sand_calculation (side_length : ℝ) (sand_weight_per_section : ℝ) (area_per_section : ℝ) :
  side_length = 40 →
  sand_weight_per_section = 30 →
  area_per_section = 80 →
  (side_length ^ 2 / area_per_section) * sand_weight_per_section = 600 :=
by sorry

end sandbox_sand_calculation_l2991_299124


namespace find_k_l2991_299134

theorem find_k : ∃ (k : ℤ) (m : ℝ), ∀ (n : ℝ), 
  n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2 → k = 3 := by
  sorry

end find_k_l2991_299134


namespace necessary_but_not_sufficient_l2991_299186

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end necessary_but_not_sufficient_l2991_299186


namespace graph_shift_up_by_two_l2991_299125

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the transformed function g
def g (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem graph_shift_up_by_two :
  ∀ x : ℝ, g x = f x + 2 := by sorry

end graph_shift_up_by_two_l2991_299125


namespace tangent_line_proof_l2991_299106

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The line equation: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_proof :
  (∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀)) ∧  -- The line passes through a point on f(x)
  line_equation 1 1 ∧  -- The line passes through (1,1)
  (∀ (x : ℝ), f' x = (2 : ℝ)) →  -- The derivative of f(x) is 2
  ∃ (x₀ : ℝ), f x₀ = x₀ ∧ line_equation x₀ (f x₀) ∧ f' x₀ = 2 :=
by sorry

end tangent_line_proof_l2991_299106


namespace derivative_symmetry_l2991_299165

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end derivative_symmetry_l2991_299165


namespace square_sum_identity_l2991_299178

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end square_sum_identity_l2991_299178


namespace inverse_proportion_ordering_l2991_299114

/-- Given points A, B, and C on the inverse proportion function y = 3/x,
    prove that y₂ < y₁ < y₃ -/
theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end inverse_proportion_ordering_l2991_299114


namespace inequality_solution_set_l2991_299189

theorem inequality_solution_set :
  {x : ℝ | 4 + 2*x > -6} = {x : ℝ | x > -5} := by
  sorry

end inequality_solution_set_l2991_299189


namespace perimeter_of_quarter_circle_bounded_square_l2991_299171

/-- The perimeter of a region bounded by quarter-circle arcs constructed on each side of a square --/
theorem perimeter_of_quarter_circle_bounded_square (s : ℝ) (h : s = 4 / Real.pi) :
  4 * (Real.pi * s / 4) = 4 := by
  sorry

end perimeter_of_quarter_circle_bounded_square_l2991_299171


namespace expression_value_l2991_299176

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 5 - 2*x + 4*y = -1 := by
  sorry

end expression_value_l2991_299176


namespace soda_difference_l2991_299122

theorem soda_difference (diet_soda : ℕ) (regular_soda : ℕ) 
  (h1 : diet_soda = 19) (h2 : regular_soda = 60) : 
  regular_soda - diet_soda = 41 := by
  sorry

end soda_difference_l2991_299122


namespace problem_1_l2991_299116

theorem problem_1 : (1/2)⁻¹ - Real.tan (π/4) + |Real.sqrt 2 - 1| = Real.sqrt 2 := by
  sorry

end problem_1_l2991_299116


namespace grass_seed_problem_l2991_299120

/-- Represents the cost and weight of a bag of grass seed -/
structure SeedBag where
  weight : Nat
  cost : Rat

/-- Represents a purchase of grass seed -/
structure Purchase where
  bags : List SeedBag
  totalWeight : Nat
  totalCost : Rat

def validPurchase (p : Purchase) : Prop :=
  p.totalWeight ≥ 65 ∧ p.totalWeight ≤ 80

def optimalPurchase (p : Purchase) : Prop :=
  validPurchase p ∧ p.totalCost = 98.75

/-- The theorem to be proved -/
theorem grass_seed_problem :
  ∃ (cost_5lb : Rat),
    let bag_5lb : SeedBag := ⟨5, cost_5lb⟩
    let bag_10lb : SeedBag := ⟨10, 20.40⟩
    let bag_25lb : SeedBag := ⟨25, 32.25⟩
    ∃ (p : Purchase),
      optimalPurchase p ∧
      bag_5lb ∈ p.bags ∧
      cost_5lb = 2.00 :=
sorry

end grass_seed_problem_l2991_299120


namespace sqrt_equation_solution_l2991_299126

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 6) = 10 → x = 106 := by
  sorry

end sqrt_equation_solution_l2991_299126


namespace smith_family_mean_age_l2991_299113

def smith_family_ages : List ℕ := [5, 5, 5, 12, 13, 16]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 9.33 := by
  sorry

end smith_family_mean_age_l2991_299113


namespace perpendicular_line_through_point_l2991_299119

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point that the line must pass through
def point : ℝ × ℝ := (1, 3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- State the theorem
theorem perpendicular_line_through_point :
  (perpendicular_line point.1 point.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (y - point.2) = -(x - point.1) * (1 / (2 : ℝ))) :=
sorry

end perpendicular_line_through_point_l2991_299119


namespace inequality_equivalence_l2991_299162

theorem inequality_equivalence (x : ℝ) : x / 3 - 2 < 0 ↔ x < 6 := by sorry

end inequality_equivalence_l2991_299162


namespace geometric_sequence_first_term_l2991_299197

theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 16 = c * r ∧ 32 = 16 * r) → a = 2 := by
  sorry

end geometric_sequence_first_term_l2991_299197


namespace system_solution_l2991_299157

theorem system_solution (p q u v : ℝ) : 
  (p * u + q * v = 2 * (p^2 - q^2)) ∧ 
  (v / (p - q) - u / (p + q) = (p^2 + q^2) / (p * q)) →
  ((p * q * (p^2 - q^2) ≠ 0 ∧ q ≠ 1 + Real.sqrt 2 ∧ q ≠ 1 - Real.sqrt 2) →
    (u = (p^2 - q^2) / p ∧ v = (p^2 - q^2) / q)) ∧
  ((u ≠ 0 ∧ v ≠ 0 ∧ u^2 ≠ v^2) →
    (p = u * v^2 / (v^2 - u^2) ∧ q = u^2 * v / (v^2 - u^2))) :=
by sorry

end system_solution_l2991_299157


namespace median_mean_difference_l2991_299184

/-- The distribution of scores on an algebra quiz -/
structure ScoreDistribution where
  score_70 : ℝ
  score_80 : ℝ
  score_90 : ℝ
  score_100 : ℝ

/-- The properties of the score distribution -/
def valid_distribution (d : ScoreDistribution) : Prop :=
  d.score_70 = 0.1 ∧
  d.score_80 = 0.35 ∧
  d.score_90 = 0.3 ∧
  d.score_100 = 0.25 ∧
  d.score_70 + d.score_80 + d.score_90 + d.score_100 = 1

/-- Calculate the mean score -/
def mean_score (d : ScoreDistribution) : ℝ :=
  70 * d.score_70 + 80 * d.score_80 + 90 * d.score_90 + 100 * d.score_100

/-- The median score -/
def median_score : ℝ := 90

/-- The main theorem: the difference between median and mean is 3 -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h : valid_distribution d) : median_score - mean_score d = 3 := by
  sorry

end median_mean_difference_l2991_299184


namespace favorite_toy_change_probability_l2991_299180

def toy_count : ℕ := 10
def min_price : ℚ := 1/2
def max_price : ℚ := 5
def price_increment : ℚ := 1/2
def initial_quarters : ℕ := 10
def favorite_toy_price : ℚ := 9/2

def toy_prices : List ℚ := 
  List.range toy_count |>.map (λ i => max_price - i * price_increment)

theorem favorite_toy_change_probability :
  let total_sequences := toy_count.factorial
  let favorable_sequences := (toy_count - 1).factorial + (toy_count - 2).factorial
  (1 : ℚ) - (favorable_sequences : ℚ) / total_sequences = 8/9 :=
sorry

end favorite_toy_change_probability_l2991_299180


namespace stock_market_value_l2991_299123

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
def market_value (dividend_rate : ℚ) (yield : ℚ) (face_value : ℚ) : ℚ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 13% stock yielding 8% with a face value of $100 has a market value of $162.50 -/
theorem stock_market_value :
  let dividend_rate : ℚ := 13 / 100
  let yield : ℚ := 8 / 100
  let face_value : ℚ := 100
  market_value dividend_rate yield face_value = 162.5 := by
  sorry

#eval market_value (13/100) (8/100) 100

end stock_market_value_l2991_299123


namespace minimum_value_implies_m_l2991_299105

noncomputable def f (x m : ℝ) : ℝ := 2 * x * Real.log (2 * x - 1) - Real.log (2 * x - 1) - m * x + Real.exp (-1)

theorem minimum_value_implies_m (h : ∀ x ∈ Set.Icc 1 (3/2), f x m ≥ -4 + Real.exp (-1)) 
  (h_min : ∃ x ∈ Set.Icc 1 (3/2), f x m = -4 + Real.exp (-1)) : 
  m = 4/3 * Real.log 2 + 8/3 :=
sorry

end minimum_value_implies_m_l2991_299105


namespace power_of_three_equality_l2991_299164

theorem power_of_three_equality (n : ℕ) : 3^n = 27 * 9^2 * (81^3) / 3^4 → n = 15 := by
  sorry

end power_of_three_equality_l2991_299164


namespace tan_half_sum_of_angles_l2991_299133

theorem tan_half_sum_of_angles (x y : Real) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end tan_half_sum_of_angles_l2991_299133


namespace total_weight_compounds_l2991_299146

/-- The atomic mass of Nitrogen in g/mol -/
def mass_N : ℝ := 14.01

/-- The atomic mass of Hydrogen in g/mol -/
def mass_H : ℝ := 1.01

/-- The atomic mass of Bromine in g/mol -/
def mass_Br : ℝ := 79.90

/-- The atomic mass of Magnesium in g/mol -/
def mass_Mg : ℝ := 24.31

/-- The atomic mass of Chlorine in g/mol -/
def mass_Cl : ℝ := 35.45

/-- The molar mass of Ammonium Bromide (NH4Br) in g/mol -/
def molar_mass_NH4Br : ℝ := mass_N + 4 * mass_H + mass_Br

/-- The molar mass of Magnesium Chloride (MgCl2) in g/mol -/
def molar_mass_MgCl2 : ℝ := mass_Mg + 2 * mass_Cl

/-- The number of moles of Ammonium Bromide -/
def moles_NH4Br : ℝ := 3.72

/-- The number of moles of Magnesium Chloride -/
def moles_MgCl2 : ℝ := 2.45

theorem total_weight_compounds : 
  moles_NH4Br * molar_mass_NH4Br + moles_MgCl2 * molar_mass_MgCl2 = 597.64 := by
  sorry

end total_weight_compounds_l2991_299146


namespace parabola_hyperbola_intersection_l2991_299140

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter p of the parabola is equal to 1. -/
theorem parabola_hyperbola_intersection (p a b : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → a > 0 → b > 0 → x₀ ≠ 0 →
  y₀^2 = 2 * p * x₀ →  -- Point A satisfies parabola equation
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- Point A satisfies hyperbola equation
  y₀ = 2 * x₀ →  -- Point A is on the asymptote y = 2x
  (x₀ - 0)^2 + y₀^2 = p^4 →  -- Distance from A to parabola's axis of symmetry is p²
  (a^2 + b^2) / a^2 = 5 →  -- Eccentricity of hyperbola is √5
  p = 1 := by
  sorry

end parabola_hyperbola_intersection_l2991_299140


namespace bruce_mango_purchase_l2991_299136

def bruce_purchase (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) (total_paid : ℕ) : ℕ :=
  let grape_cost := grape_quantity * grape_price
  let mango_cost := total_paid - grape_cost
  mango_cost / mango_price

theorem bruce_mango_purchase :
  bruce_purchase 7 70 55 985 = 9 := by
  sorry

end bruce_mango_purchase_l2991_299136


namespace sin_cos_sum_implies_tan_value_l2991_299161

theorem sin_cos_sum_implies_tan_value (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x + Real.cos x = 3 * Real.sqrt 2 / 5) : 
  (1 - Real.cos (2 * x)) / Real.sin (2 * x) = -7 := by
  sorry

end sin_cos_sum_implies_tan_value_l2991_299161


namespace cubic_root_sum_product_l2991_299115

theorem cubic_root_sum_product (a b : ℝ) : 
  (a^3 - 4*a^2 - a + 4 = 0) → 
  (b^3 - 4*b^2 - b + 4 = 0) → 
  (a ≠ b) →
  (a + b + a*b = -1) := by
sorry

end cubic_root_sum_product_l2991_299115


namespace common_chord_length_l2991_299108

/-- The length of the common chord of two intersecting circles -/
theorem common_chord_length (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 10*y - 24 = 0) →
  (x^2 + y^2 + 2*x + 2*y - 8 = 0) →
  ∃ (l : ℝ), l = 2 * Real.sqrt 5 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2 + y1^2 - 2*x1 + 10*y1 - 24 = 0) ∧
      (x1^2 + y1^2 + 2*x1 + 2*y1 - 8 = 0) ∧
      (x2^2 + y2^2 - 2*x2 + 10*y2 - 24 = 0) ∧
      (x2^2 + y2^2 + 2*x2 + 2*y2 - 8 = 0) ∧
      l^2 = (x2 - x1)^2 + (y2 - y1)^2) :=
by
  sorry

end common_chord_length_l2991_299108


namespace cut_cube_theorem_l2991_299187

/-- Given a cube cut into equal smaller cubes, this function calculates
    the total number of smaller cubes created. -/
def total_smaller_cubes (n : ℕ) : ℕ := (n + 1)^3

/-- This function calculates the number of smaller cubes painted on exactly 2 faces. -/
def cubes_with_two_painted_faces (n : ℕ) : ℕ := 12 * (n - 1)

/-- Theorem stating that when a cube is cut such that 12 smaller cubes are painted
    on exactly 2 faces, the total number of smaller cubes is 27. -/
theorem cut_cube_theorem :
  ∃ n : ℕ, cubes_with_two_painted_faces n = 12 ∧ total_smaller_cubes n = 27 :=
sorry

end cut_cube_theorem_l2991_299187


namespace f_at_one_l2991_299112

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end f_at_one_l2991_299112


namespace g_difference_l2991_299138

/-- A linear function g satisfying g(d+2) - g(d) = 8 for all real numbers d -/
def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
  (∀ d : ℝ, g (d + 2) - g d = 8)

theorem g_difference (g : ℝ → ℝ) (h : g_property g) : g 1 - g 7 = -24 := by
  sorry

end g_difference_l2991_299138


namespace doubling_function_m_range_l2991_299188

/-- A function f is a doubling function if there exists an interval [a, b] such that f([a, b]) = [2a, 2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ Set.image f (Set.Icc a b) = Set.Icc (2*a) (2*b)

/-- The main theorem stating that if ln(e^x + m) is a doubling function, then m is in the open interval (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, DoublingFunction (fun x ↦ Real.log (Real.exp x + m)) → m ∈ Set.Ioo (-1/4) 0 :=
by sorry

end doubling_function_m_range_l2991_299188


namespace sports_club_membership_l2991_299199

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 27)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_both : both = 11) :
  total - (badminton + tennis - both) = 2 :=
by sorry

end sports_club_membership_l2991_299199


namespace angle_in_square_l2991_299195

/-- In a square ABCD with a segment CE, if CE forms angles of 7α and 8α with the sides of the square, then α = 9°. -/
theorem angle_in_square (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by sorry

end angle_in_square_l2991_299195


namespace hotel_expenditure_l2991_299141

/-- The total expenditure of a group of men, where most spend a fixed amount and one spends more than the average -/
def total_expenditure (n : ℕ) (m : ℕ) (fixed_spend : ℚ) (extra_spend : ℚ) : ℚ :=
  let avg := (m * fixed_spend + ((m * fixed_spend + extra_spend) / n)) / n
  n * avg

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem hotel_expenditure :
  round_to_nearest (total_expenditure 9 8 3 5) = 33 := by
  sorry

end hotel_expenditure_l2991_299141


namespace largest_n_divisibility_l2991_299196

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 246 → ¬(∃ k : ℤ, n^3 + 150 = k * (n + 12)) ∧
  ∃ k : ℤ, 246^3 + 150 = k * (246 + 12) :=
by sorry

end largest_n_divisibility_l2991_299196


namespace shaded_area_is_9_sqrt_3_l2991_299151

-- Define the square
structure Square where
  side : ℝ
  height : ℝ
  bottomRight : ℝ × ℝ

-- Define the equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  height : ℝ
  bottomLeft : ℝ × ℝ

-- Define the problem setup
def problemSetup (s : Square) (t : EquilateralTriangle) : Prop :=
  s.side = 14 ∧
  t.side = 18 ∧
  s.height = t.height ∧
  s.bottomRight = (14, 0) ∧
  t.bottomLeft = (14, 0)

-- Define the shaded area
def shadedArea (s : Square) (t : EquilateralTriangle) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_9_sqrt_3 (s : Square) (t : EquilateralTriangle) :
  problemSetup s t → shadedArea s t = 9 * Real.sqrt 3 := by sorry

end shaded_area_is_9_sqrt_3_l2991_299151


namespace quadratic_symmetry_point_l2991_299182

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*(a-1)

theorem quadratic_symmetry_point (a : ℝ) :
  (∀ x ≤ 4, (f_derivative a x ≤ 0)) ∧
  (∀ x ≥ 4, (f_derivative a x ≥ 0)) →
  a = -3 := by sorry

end quadratic_symmetry_point_l2991_299182


namespace room_area_in_square_yards_l2991_299174

/-- Proves that the area of a 15 ft by 10 ft rectangular room is 16.67 square yards -/
theorem room_area_in_square_yards :
  let length : ℝ := 15
  let width : ℝ := 10
  let sq_feet_per_sq_yard : ℝ := 9
  let area_sq_feet : ℝ := length * width
  let area_sq_yards : ℝ := area_sq_feet / sq_feet_per_sq_yard
  area_sq_yards = 16.67 := by sorry

end room_area_in_square_yards_l2991_299174


namespace play_role_assignments_l2991_299109

def number_of_assignments (men women : ℕ) (specific_male_roles specific_female_roles either_gender_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - 2) either_gender_roles)

theorem play_role_assignments :
  number_of_assignments 6 7 1 1 4 = 13860 := by sorry

end play_role_assignments_l2991_299109


namespace kendra_spelling_goals_l2991_299170

-- Define constants
def words_per_week : ℕ := 12
def first_goal : ℕ := 60
def second_goal : ℕ := 100
def reward_threshold : ℕ := 20
def words_learned : ℕ := 36
def weeks_to_birthday : ℕ := 3
def weeks_to_competition : ℕ := 6

-- Define the theorem
theorem kendra_spelling_goals (target : ℕ) :
  (target ≥ reward_threshold) ∧
  (target * weeks_to_birthday + words_learned ≥ first_goal) ∧
  (target * weeks_to_competition + words_learned ≥ second_goal) ↔
  target = reward_threshold :=
by sorry

end kendra_spelling_goals_l2991_299170


namespace population_reproduction_after_development_l2991_299111

/-- Represents the types of population reproduction --/
inductive PopulationReproductionType
  | Primitive
  | Traditional
  | TransitionToModern
  | Modern

/-- Represents the state of society after a major development of productive forces --/
structure SocietyState where
  productiveForcesDeveloped : Bool
  materialWealthIncreased : Bool
  populationGrowthRapid : Bool
  healthCareImproved : Bool
  mortalityRatesDecreased : Bool

/-- Determines the type of population reproduction based on the society state --/
def determinePopulationReproductionType (state : SocietyState) : PopulationReproductionType :=
  if state.productiveForcesDeveloped ∧
     state.materialWealthIncreased ∧
     state.populationGrowthRapid ∧
     state.healthCareImproved ∧
     state.mortalityRatesDecreased
  then PopulationReproductionType.Traditional
  else PopulationReproductionType.Primitive

/-- Theorem stating that after the first major development of productive forces, 
    the population reproduction type was Traditional --/
theorem population_reproduction_after_development 
  (state : SocietyState) 
  (h1 : state.productiveForcesDeveloped)
  (h2 : state.materialWealthIncreased)
  (h3 : state.populationGrowthRapid)
  (h4 : state.healthCareImproved)
  (h5 : state.mortalityRatesDecreased) :
  determinePopulationReproductionType state = PopulationReproductionType.Traditional := by
  sorry

end population_reproduction_after_development_l2991_299111


namespace quadratic_equal_roots_l2991_299150

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = -2 + 6 * Real.sqrt 5 ∨ m = -2 - 6 * Real.sqrt 5) :=
sorry

end quadratic_equal_roots_l2991_299150


namespace solve_rain_problem_l2991_299177

def rain_problem (x : ℝ) : Prop :=
  let monday_total := x + 1
  let tuesday := 2 * monday_total
  let wednesday := 0
  let thursday := 1
  let friday := monday_total + tuesday + wednesday + thursday
  let total_rain := monday_total + tuesday + wednesday + thursday + friday
  let daily_average := 4
  total_rain = 7 * daily_average ∧ x > 0

theorem solve_rain_problem :
  ∃ x : ℝ, rain_problem x ∧ x = 10 / 3 := by
  sorry

end solve_rain_problem_l2991_299177


namespace dog_fruits_total_l2991_299185

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ
  cherries : ℕ

/-- The conditions of the problem and the theorem to prove -/
theorem dog_fruits_total (df : DogFruits) : 
  df.apples = 3 * df.blueberries →
  df.blueberries = (3 * df.bonnies) / 4 →
  df.cherries = 5 * df.apples →
  df.bonnies = 60 →
  df.apples + df.blueberries + df.bonnies + df.cherries = 915 := by
  sorry

#check dog_fruits_total

end dog_fruits_total_l2991_299185


namespace possible_m_values_l2991_299148

theorem possible_m_values (m : ℝ) : 
  (2 ∈ ({m - 1, 2 * m, m^2 - 1} : Set ℝ)) → 
  (m ∈ ({3, Real.sqrt 3, -Real.sqrt 3} : Set ℝ)) := by
sorry

end possible_m_values_l2991_299148


namespace new_average_weight_l2991_299168

def original_players : ℕ := 7
def original_average_weight : ℝ := 112
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight := original_players * original_average_weight
  let total_new_weight := total_original_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  (total_new_weight / new_total_players : ℝ) = 106 := by
  sorry

end new_average_weight_l2991_299168


namespace max_value_of_expression_l2991_299190

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2*(b+c) + b^2*(c+a) + c^2*(a+b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) :=
by sorry

end max_value_of_expression_l2991_299190


namespace product_upper_bound_l2991_299118

theorem product_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) : a * b ≤ 4 := by
  sorry

end product_upper_bound_l2991_299118


namespace ellipse_foci_distance_l2991_299137

/-- The distance between the foci of an ellipse with equation 
    √((x-4)² + (y-5)²) + √((x+6)² + (y-9)²) = 24 is equal to 2√29 -/
theorem ellipse_foci_distance : 
  let ellipse_eq (x y : ℝ) := Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24
  ∃ f₁ f₂ : ℝ × ℝ, (∀ x y : ℝ, ellipse_eq x y → Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 24) ∧
             Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 2 * Real.sqrt 29 :=
by sorry

end ellipse_foci_distance_l2991_299137


namespace area_ratio_is_one_l2991_299103

/-- Theorem: The ratio of the areas of rectangles M and N is 1 -/
theorem area_ratio_is_one (a b x y : ℝ) : a > 0 → b > 0 → x > 0 → y > 0 → 
  b * x + a * y = a * b → (x * y) / ((a - x) * (b - y)) = 1 := by
  sorry

end area_ratio_is_one_l2991_299103


namespace sum_in_interval_l2991_299100

theorem sum_in_interval : 
  let sum := 4 + 3/8 + 5 + 3/4 + 7 + 2/25
  17 < sum ∧ sum < 18 := by
  sorry

end sum_in_interval_l2991_299100


namespace performance_stability_comparison_l2991_299110

/-- Represents the variance of a student's scores -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines when one performance is more stable than another based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value > b.value

theorem performance_stability_comparison
  (S_A : StudentVariance)
  (S_B : StudentVariance)
  (h_A : S_A.value = 0.2)
  (h_B : S_B.value = 0.09) :
  more_stable S_A S_B = false :=
by sorry

end performance_stability_comparison_l2991_299110


namespace stamp_collection_ratio_l2991_299144

theorem stamp_collection_ratio : 
  ∀ (tom_original mike_gift harry_gift tom_final : ℕ),
    tom_original = 3000 →
    mike_gift = 17 →
    ∃ k : ℕ, harry_gift = k * mike_gift + 10 →
    tom_final = tom_original + mike_gift + harry_gift →
    tom_final = 3061 →
    harry_gift / mike_gift = 44 / 17 := by
  sorry

end stamp_collection_ratio_l2991_299144


namespace greatest_common_measure_l2991_299127

theorem greatest_common_measure (a b c : ℕ) (ha : a = 18000) (hb : b = 50000) (hc : c = 1520) :
  Nat.gcd a (Nat.gcd b c) = 40 := by
  sorry

end greatest_common_measure_l2991_299127


namespace bijection_probability_l2991_299175

-- Define sets A and B
def A : Set (Fin 2) := Set.univ
def B : Set (Fin 3) := Set.univ

-- Define the total number of mappings from A to B
def total_mappings : ℕ := 3^2

-- Define the number of bijective mappings from A to B
def bijective_mappings : ℕ := 3 * 2

-- Define the probability of a random mapping being bijective
def prob_bijective : ℚ := bijective_mappings / total_mappings

-- Theorem statement
theorem bijection_probability :
  prob_bijective = 2/3 := by sorry

end bijection_probability_l2991_299175


namespace sin_20_cos_10_minus_cos_160_sin_10_l2991_299129

theorem sin_20_cos_10_minus_cos_160_sin_10 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_20_cos_10_minus_cos_160_sin_10_l2991_299129


namespace floor_ceil_sum_seven_l2991_299192

theorem floor_ceil_sum_seven (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end floor_ceil_sum_seven_l2991_299192


namespace triangle_identities_l2991_299169

theorem triangle_identities (a b c α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
sorry

end triangle_identities_l2991_299169


namespace function_identity_l2991_299102

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end function_identity_l2991_299102


namespace same_terminal_side_angle_with_same_terminal_side_l2991_299135

theorem same_terminal_side (θ₁ θ₂ : Real) : 
  ∃ k : Int, θ₂ = θ₁ + 2 * π * k → 
  θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * k :=
by sorry

end same_terminal_side_angle_with_same_terminal_side_l2991_299135


namespace volleyball_team_math_players_l2991_299132

theorem volleyball_team_math_players 
  (total_players : ℕ) 
  (physics_players : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_players = 15)
  (h2 : physics_players = 10)
  (h3 : both_subjects = 4)
  (h4 : physics_players ≤ total_players)
  (h5 : both_subjects ≤ physics_players)
  (h6 : ∀ p, p ∈ (Finset.range total_players) → 
    (p ∈ (Finset.range physics_players) ∨ 
     p ∈ (Finset.range (total_players - physics_players + both_subjects)))) :
  total_players - physics_players + both_subjects = 9 := by
sorry

end volleyball_team_math_players_l2991_299132


namespace domain_of_g_l2991_299191

-- Define the function f with domain (-3, 6)
def f : {x : ℝ // -3 < x ∧ x < 6} → ℝ := sorry

-- Define the function g(x) = f(2x)
def g (x : ℝ) : ℝ := f ⟨2*x, sorry⟩

-- Theorem statement
theorem domain_of_g :
  ∀ x : ℝ, (∃ y : ℝ, g x = y) ↔ -3/2 < x ∧ x < 3 :=
sorry

end domain_of_g_l2991_299191


namespace puzzle_assembly_time_l2991_299193

-- Define the number of pieces in the puzzle
def puzzle_pieces : ℕ := 121

-- Define the time it takes to assemble the puzzle with the original method
def original_time : ℕ := 120

-- Define the function for the original assembly method (2 pieces per minute)
def original_assembly (t : ℕ) : ℕ := puzzle_pieces - t

-- Define the function for the new assembly method (3 pieces per minute)
def new_assembly (t : ℕ) : ℕ := puzzle_pieces - 2 * t

-- State the theorem
theorem puzzle_assembly_time :
  ∃ (new_time : ℕ), 
    (original_assembly original_time = 1) ∧ 
    (new_assembly new_time = 1) ∧ 
    (new_time = original_time / 2) := by
  sorry

end puzzle_assembly_time_l2991_299193


namespace unique_phone_number_l2991_299131

-- Define the set of available digits
def available_digits : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

-- Define a function to check if a list of digits is valid
def valid_phone_number (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = available_digits ∧
  digits.Sorted (·<·)

-- Theorem statement
theorem unique_phone_number :
  ∃! digits : List Nat, valid_phone_number digits :=
sorry

end unique_phone_number_l2991_299131


namespace chess_tournament_games_l2991_299117

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays
    four times with each opponent, the total number of games is 1740 -/
theorem chess_tournament_games :
  tournament_games 30 4 = 1740 := by
  sorry

#eval tournament_games 30 4

end chess_tournament_games_l2991_299117


namespace simplify_expression_l2991_299107

theorem simplify_expression (x y : ℝ) : 7*x + 9*y + 3 - x + 12*y + 15 = 6*x + 21*y + 18 := by
  sorry

end simplify_expression_l2991_299107


namespace smallest_multiple_of_4_to_8_exists_840_multiple_l2991_299149

theorem smallest_multiple_of_4_to_8 : ∀ n : ℕ, n > 0 → (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 :=
by
  sorry

theorem exists_840_multiple : (4 ∣ 840) ∧ (5 ∣ 840) ∧ (6 ∣ 840) ∧ (7 ∣ 840) ∧ (8 ∣ 840) :=
by
  sorry

end smallest_multiple_of_4_to_8_exists_840_multiple_l2991_299149


namespace distribute_five_into_three_l2991_299101

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_five_into_three : distribute 5 3 = 36 := by sorry

end distribute_five_into_three_l2991_299101


namespace rational_x_y_l2991_299173

theorem rational_x_y (x y : ℝ) 
  (h : ∀ (p q : ℕ), Prime p → Prime q → Odd p → Odd q → p ≠ q → 
    ∃ (r : ℚ), (x^p + y^q : ℝ) = (r : ℝ)) : 
  ∃ (a b : ℚ), (x = (a : ℝ) ∧ y = (b : ℝ)) := by
sorry

end rational_x_y_l2991_299173
