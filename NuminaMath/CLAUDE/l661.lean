import Mathlib

namespace rationalize_denominator_l661_66180

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
sorry

end rationalize_denominator_l661_66180


namespace quadratic_zeros_imply_range_bound_l661_66102

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_zeros_imply_range_bound (b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    quadratic_function b c x₁ = 0 ∧ quadratic_function b c x₂ = 0) →
  0 < (1 + b) * c + c^2 ∧ (1 + b) * c + c^2 < 1/16 :=
by sorry

end quadratic_zeros_imply_range_bound_l661_66102


namespace ring_cost_calculation_l661_66119

/-- The cost of a single ring given the total sales and necklace price -/
def ring_cost (total_sales necklace_price : ℕ) (num_necklaces num_rings : ℕ) : ℕ :=
  (total_sales - necklace_price * num_necklaces) / num_rings

theorem ring_cost_calculation (total_sales necklace_price : ℕ) 
  (h1 : total_sales = 80)
  (h2 : necklace_price = 12)
  (h3 : ring_cost total_sales necklace_price 4 8 = 4) : 
  ∃ (x : ℕ), x = ring_cost 80 12 4 8 ∧ x = 4 := by
  sorry

end ring_cost_calculation_l661_66119


namespace problem_solution_l661_66153

theorem problem_solution :
  ∀ x y : ℕ,
    x > 0 → y > 0 →
    x < 15 → y < 15 →
    x + y + x * y = 49 →
    x + y = 13 := by
  sorry

end problem_solution_l661_66153


namespace binary_263_ones_minus_zeros_l661_66170

def binary_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem binary_263_ones_minus_zeros :
  let bin_263 := binary_representation 263
  let x := count_zeros bin_263
  let y := count_ones bin_263
  y - x = 0 := by sorry

end binary_263_ones_minus_zeros_l661_66170


namespace square_area_proof_l661_66166

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 24 - 2 * x) → 
  ((3 * x - 12) ^ 2 = 92.16) := by
sorry

end square_area_proof_l661_66166


namespace fedya_deposit_l661_66143

theorem fedya_deposit (k : ℕ) (h1 : k < 30) (h2 : k > 0) : 
  ∃ (n : ℕ), n * (100 - k) = 84700 ∧ n = 1100 := by
sorry

end fedya_deposit_l661_66143


namespace even_function_property_l661_66179

/-- A function f is even on an interval [-a, a] -/
def IsEvenOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-a) a, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ) (h_even : IsEvenOn f 6) (h_gt : f 3 > f 1) :
  f (-1) < f 3 := by
  sorry

end even_function_property_l661_66179


namespace dodecahedron_interior_diagonals_l661_66154

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Fin 12)
  is_pentagonal : faces → Prop
  vertex_face_incidence : vertices → faces → Prop
  three_faces_per_vertex : ∀ v : vertices, ∃! (f1 f2 f3 : faces), 
    vertex_face_incidence v f1 ∧ vertex_face_incidence v f2 ∧ vertex_face_incidence v f3 ∧ f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3

/-- An interior diagonal in a dodecahedron -/
def interior_diagonal (d : Dodecahedron) (v1 v2 : d.vertices) : Prop :=
  v1 ≠ v2 ∧ ∀ f : d.faces, ¬(d.vertex_face_incidence v1 f ∧ d.vertex_face_incidence v2 f)

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices.card * (d.vertices.card - 3)) / 2

/-- Theorem stating that a dodecahedron has 170 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  num_interior_diagonals d = 170 := by sorry

end dodecahedron_interior_diagonals_l661_66154


namespace equation_3x_eq_4y_is_linear_l661_66118

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The equation 3x = 4y is a linear equation in two variables. -/
theorem equation_3x_eq_4y_is_linear :
  IsLinearEquationInTwoVariables (fun x y => 3 * x - 4 * y) :=
sorry

end equation_3x_eq_4y_is_linear_l661_66118


namespace impossible_arrangement_l661_66187

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Returns the number of people between two positions on the table -/
def distanceBetween (table : Table) (p1 p2 : Fin 40) : Nat :=
  sorry

/-- Checks if two people have a common acquaintance -/
def haveCommonAcquaintance (table : Table) (p1 p2 : Fin 40) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement (table : Table) : 
  ¬(∀ (p1 p2 : Fin 40), 
    (distanceBetween table p1 p2 % 2 = 0 → haveCommonAcquaintance table p1 p2) ∧
    (distanceBetween table p1 p2 % 2 = 1 → ¬haveCommonAcquaintance table p1 p2)) :=
  sorry

end impossible_arrangement_l661_66187


namespace test_questions_l661_66146

theorem test_questions (sections : ℕ) (correct_answers : ℕ) (lower_bound : ℚ) (upper_bound : ℚ) :
  sections = 5 →
  correct_answers = 32 →
  lower_bound = 70/100 →
  upper_bound = 77/100 →
  ∃ (total_questions : ℕ),
    (total_questions % sections = 0) ∧
    (lower_bound < (correct_answers : ℚ) / total_questions) ∧
    ((correct_answers : ℚ) / total_questions < upper_bound) ∧
    (total_questions = 45) := by
  sorry

#check test_questions

end test_questions_l661_66146


namespace new_device_significantly_improved_l661_66188

-- Define the sample means and variances
def x_bar : ℝ := 10
def y_bar : ℝ := 10.3
def s1_squared : ℝ := 0.036
def s2_squared : ℝ := 0.04

-- Define the significant improvement criterion
def significant_improvement (x_bar y_bar s1_squared s2_squared : ℝ) : Prop :=
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / 10)

-- Theorem statement
theorem new_device_significantly_improved :
  significant_improvement x_bar y_bar s1_squared s2_squared := by
  sorry


end new_device_significantly_improved_l661_66188


namespace equation_solution_l661_66106

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end equation_solution_l661_66106


namespace min_distance_to_line_l661_66197

theorem min_distance_to_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  A = (-2, 0) →
  B = (0, 3) →
  (∀ x y, l x y ↔ x - y + 1 = 0) →
  ∃ P : ℝ × ℝ, l P.1 P.2 ∧
    (∀ Q : ℝ × ℝ, l Q.1 Q.2 → Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
                               Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤
                               Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2) +
                               Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)) ∧
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = Real.sqrt 17 :=
by sorry


end min_distance_to_line_l661_66197


namespace additional_earnings_is_correct_l661_66130

/-- Represents the company's dividend policy and earnings information -/
structure CompanyData where
  expected_earnings : ℝ
  actual_earnings : ℝ
  base_dividend_ratio : ℝ
  extra_dividend_rate : ℝ
  shares_owned : ℕ
  total_dividend_paid : ℝ

/-- Calculates the additional earnings per share that triggers the extra dividend -/
def additional_earnings (data : CompanyData) : ℝ :=
  data.actual_earnings - data.expected_earnings

/-- Theorem stating that the additional earnings per share is $0.30 -/
theorem additional_earnings_is_correct (data : CompanyData) 
  (h1 : data.expected_earnings = 0.80)
  (h2 : data.actual_earnings = 1.10)
  (h3 : data.base_dividend_ratio = 0.5)
  (h4 : data.extra_dividend_rate = 0.04)
  (h5 : data.shares_owned = 400)
  (h6 : data.total_dividend_paid = 208) :
  additional_earnings data = 0.30 := by
  sorry

#eval additional_earnings {
  expected_earnings := 0.80,
  actual_earnings := 1.10,
  base_dividend_ratio := 0.5,
  extra_dividend_rate := 0.04,
  shares_owned := 400,
  total_dividend_paid := 208
}

end additional_earnings_is_correct_l661_66130


namespace cost_price_calculation_l661_66134

/-- Given an article sold at a 30% profit with a selling price of 364,
    prove that the cost price of the article is 280. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 364)
    (h2 : profit_percentage = 0.30) : 
  ∃ (cost_price : ℝ), cost_price = 280 ∧ 
    selling_price = cost_price * (1 + profit_percentage) := by
  sorry

end cost_price_calculation_l661_66134


namespace x_plus_y_value_l661_66140

theorem x_plus_y_value (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 7)) : 
  x + y = 8.5 := by
sorry

end x_plus_y_value_l661_66140


namespace total_cost_of_pens_l661_66195

/-- The cost of a single pen in dollars -/
def cost_per_pen : ℚ := 2

/-- The number of pens -/
def number_of_pens : ℕ := 10

/-- The total cost of pens -/
def total_cost : ℚ := cost_per_pen * number_of_pens

/-- Theorem stating that the total cost of 10 pens is $20 -/
theorem total_cost_of_pens : total_cost = 20 := by sorry

end total_cost_of_pens_l661_66195


namespace correct_weight_calculation_l661_66181

theorem correct_weight_calculation (class_size : ℕ) 
  (incorrect_avg : ℚ) (misread_weight : ℚ) (correct_avg : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  misread_weight = 56 →
  correct_avg = 58.6 →
  (class_size : ℚ) * correct_avg - (class_size : ℚ) * incorrect_avg + misread_weight = 60 :=
by sorry

end correct_weight_calculation_l661_66181


namespace function_inequality_implies_parameter_range_l661_66123

/-- Given f(x) = ln x - a, if f(x) < x^2 holds for all x > 1, then a ≥ -1 -/
theorem function_inequality_implies_parameter_range (a : ℝ) :
  (∀ x > 1, Real.log x - a < x^2) →
  a ≥ -1 := by
  sorry

end function_inequality_implies_parameter_range_l661_66123


namespace r_value_when_n_is_three_l661_66162

theorem r_value_when_n_is_three :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 2^s + s
  r = 135 := by sorry

end r_value_when_n_is_three_l661_66162


namespace rhind_papyrus_fraction_decomposition_l661_66108

theorem rhind_papyrus_fraction_decomposition : 
  2 / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end rhind_papyrus_fraction_decomposition_l661_66108


namespace coffee_thermoses_count_l661_66168

-- Define the conversion factor from gallons to pints
def gallons_to_pints : ℚ := 8

-- Define the total amount of coffee in gallons
def total_coffee_gallons : ℚ := 9/2

-- Define the number of thermoses Genevieve drank
def thermoses_consumed : ℕ := 3

-- Define the amount of coffee Genevieve consumed in pints
def coffee_consumed_pints : ℕ := 6

-- Theorem to prove
theorem coffee_thermoses_count : 
  (total_coffee_gallons * gallons_to_pints) / (coffee_consumed_pints / thermoses_consumed) = 18 := by
  sorry

end coffee_thermoses_count_l661_66168


namespace solution_set_inequality_l661_66157

theorem solution_set_inequality (x : ℝ) : 
  (x^2 - |x - 1| - 1 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := by
  sorry

end solution_set_inequality_l661_66157


namespace binomial_coefficient_equality_l661_66144

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 20 (3 * x) = Nat.choose 20 (x + 4) → x = 2 ∨ x = 4 := by
  sorry

end binomial_coefficient_equality_l661_66144


namespace largest_possible_s_value_l661_66159

theorem largest_possible_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (((r - 2) * 180 : ℚ) / r) / (((s - 2) * 180 : ℚ) / s) = 101 / 97 → s ≤ 100 := by
  sorry

end largest_possible_s_value_l661_66159


namespace smallest_divisible_by_15_18_20_l661_66155

theorem smallest_divisible_by_15_18_20 : 
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by sorry

end smallest_divisible_by_15_18_20_l661_66155


namespace kevins_toad_feeding_l661_66165

/-- Given Kevin's toad feeding scenario, prove the number of worms per toad. -/
theorem kevins_toad_feeding (num_toads : ℕ) (minutes_per_worm : ℕ) (total_hours : ℕ) 
  (h1 : num_toads = 8)
  (h2 : minutes_per_worm = 15)
  (h3 : total_hours = 6) :
  (total_hours * 60) / minutes_per_worm / num_toads = 3 := by
  sorry

#check kevins_toad_feeding

end kevins_toad_feeding_l661_66165


namespace train_length_l661_66194

/-- Given a train traveling at 90 kmph and crossing a pole in 4 seconds, its length is 100 meters. -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed_kmph = 90 → 
  crossing_time = 4 → 
  train_length = speed_kmph * (1000 / 3600) * crossing_time →
  train_length = 100 := by
  sorry

#check train_length

end train_length_l661_66194


namespace minimum_value_and_tangent_line_l661_66161

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + 1 / (a * Real.exp x) + b

theorem minimum_value_and_tangent_line (a b : ℝ) (ha : a > 0) :
  (∀ x ≥ 0, f a b x ≥ (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  (∃ x ≥ 0, f a b x = (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  ((f a b 2 = 3 ∧ (deriv (f a b)) 2 = 3/2) → a = 2 / Real.exp 2 ∧ b = 1/2) :=
sorry

end minimum_value_and_tangent_line_l661_66161


namespace siblings_age_sum_l661_66189

theorem siblings_age_sum (ages : Fin 6 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ ages i) 
  (h_mean : (Finset.univ.sum ages) / 6 = 10)
  (h_median : (ages (Fin.mk 2 (by norm_num)) + ages (Fin.mk 3 (by norm_num))) / 2 = 12) :
  ages 0 + ages 5 = 12 := by
sorry

end siblings_age_sum_l661_66189


namespace large_circle_diameter_is_32_l661_66133

/-- Represents the arrangement of circles as described in the problem -/
structure CircleArrangement where
  small_circle_radius : ℝ
  num_small_circles : ℕ
  num_layers : ℕ

/-- The specific arrangement described in the problem -/
def problem_arrangement : CircleArrangement :=
  { small_circle_radius := 4
  , num_small_circles := 8
  , num_layers := 2 }

/-- The diameter of the large circle in the arrangement -/
def large_circle_diameter (ca : CircleArrangement) : ℝ := 32

/-- Theorem stating that the diameter of the large circle in the problem arrangement is 32 units -/
theorem large_circle_diameter_is_32 :
  large_circle_diameter problem_arrangement = 32 := by
  sorry

end large_circle_diameter_is_32_l661_66133


namespace cube_preserves_order_l661_66116

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_order_l661_66116


namespace honey_container_size_l661_66135

/-- The number of ounces in Tabitha's honey container -/
def honey_container_ounces : ℕ :=
  let servings_per_cup : ℕ := 1
  let cups_per_night : ℕ := 2
  let servings_per_ounce : ℕ := 6
  let nights_honey_lasts : ℕ := 48
  (servings_per_cup * cups_per_night * nights_honey_lasts) / servings_per_ounce

theorem honey_container_size :
  honey_container_ounces = 16 := by
  sorry

end honey_container_size_l661_66135


namespace three_digit_sum_l661_66174

theorem three_digit_sum (A B : ℕ) : 
  A < 10 → 
  B < 10 → 
  100 ≤ 14 * 10 + A → 
  14 * 10 + A < 1000 → 
  100 ≤ 100 * B + 73 → 
  100 * B + 73 < 1000 → 
  14 * 10 + A + 100 * B + 73 = 418 → 
  A = 5 := by sorry

end three_digit_sum_l661_66174


namespace josie_safari_count_l661_66184

/-- The total number of animals Josie counted on safari -/
def total_animals (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

/-- Theorem stating the total number of animals Josie counted -/
theorem josie_safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards giraffes lions elephants = 1308 :=
by
  sorry

end josie_safari_count_l661_66184


namespace unit_circle_arc_angle_l661_66177

/-- The central angle (in radians) corresponding to an arc of length 1 in a unit circle is 1. -/
theorem unit_circle_arc_angle (θ : ℝ) : θ = 1 := by
  sorry

end unit_circle_arc_angle_l661_66177


namespace inequality_proof_l661_66105

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^1999 + b^2000 ≥ a^2000 + b^2001) : 
  a^2000 + b^2000 ≤ 2 := by sorry

end inequality_proof_l661_66105


namespace economy_to_luxury_ratio_l661_66113

/-- Represents the ratio between two quantities -/
structure Ratio where
  antecedent : ℕ
  consequent : ℕ

/-- Represents the inventory of a car dealership -/
structure CarInventory where
  economy_to_suv : Ratio
  luxury_to_suv : Ratio

theorem economy_to_luxury_ratio (inventory : CarInventory) 
  (h1 : inventory.economy_to_suv = Ratio.mk 4 1)
  (h2 : inventory.luxury_to_suv = Ratio.mk 8 1) :
  Ratio.mk 1 2 = 
    Ratio.mk 
      (inventory.economy_to_suv.antecedent * inventory.luxury_to_suv.consequent)
      (inventory.economy_to_suv.consequent * inventory.luxury_to_suv.antecedent) :=
by sorry

end economy_to_luxury_ratio_l661_66113


namespace neighborhood_cable_cost_l661_66171

/-- Calculates the total cost of power cable for a neighborhood grid --/
theorem neighborhood_cable_cost
  (ew_streets : ℕ) (ew_length : ℕ)
  (ns_streets : ℕ) (ns_length : ℕ)
  (cable_per_street : ℕ) (cable_cost : ℕ) :
  ew_streets = 18 →
  ew_length = 2 →
  ns_streets = 10 →
  ns_length = 4 →
  cable_per_street = 5 →
  cable_cost = 2000 →
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_street * cable_cost = 760000 :=
by sorry

end neighborhood_cable_cost_l661_66171


namespace reuschles_theorem_l661_66150

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁ on the sides of triangle ABC
variable (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the condition that AA₁, BB₁, CC₁ intersect at a single point
def lines_concurrent (A B C A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- Define the circumcircle of triangle A₁B₁C₁
def circumcircle (A₁ B₁ C₁ : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define points A₂, B₂, C₂ as the second intersection points
def second_intersection (A B C A₁ B₁ C₁ : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem statement
theorem reuschles_theorem (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  lines_concurrent A B C A₁ B₁ C₁ →
  let (A₂, B₂, C₂) := second_intersection A B C A₁ B₁ C₁
  lines_concurrent A B C A₂ B₂ C₂ := by sorry

end reuschles_theorem_l661_66150


namespace total_distance_walked_l661_66182

def distance_school_to_david : ℝ := 0.2
def distance_david_to_home : ℝ := 0.7

theorem total_distance_walked : 
  distance_school_to_david + distance_david_to_home = 0.9 := by sorry

end total_distance_walked_l661_66182


namespace cricket_team_size_l661_66131

-- Define the number of team members
variable (n : ℕ)

-- Define the average age of the team
def team_average : ℝ := 25

-- Define the wicket keeper's age
def wicket_keeper_age : ℝ := team_average + 3

-- Define the average age of remaining players after excluding two members
def remaining_average : ℝ := team_average - 1

-- Define the total age of the team
def total_age : ℝ := n * team_average

-- Define the total age of remaining players
def remaining_total_age : ℝ := (n - 2) * remaining_average

-- Define the total age of the two excluded members
def excluded_total_age : ℝ := wicket_keeper_age + team_average

-- Theorem stating that the number of team members is 5
theorem cricket_team_size : n = 5 := by
  sorry

end cricket_team_size_l661_66131


namespace existence_of_special_set_l661_66120

theorem existence_of_special_set :
  ∃ (S : Finset ℕ), 
    Finset.card S = 1998 ∧ 
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → (a * b) % ((a - b) ^ 2) = 0 :=
sorry

end existence_of_special_set_l661_66120


namespace inequality_proof_l661_66136

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end inequality_proof_l661_66136


namespace min_abs_phi_l661_66142

theorem min_abs_phi (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) : 
  (∀ x, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (x + π) + φ)) →
  (∀ x, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (2 * π / 3 - x) + φ)) →
  ∃ k : ℤ, |φ + k * π| = π / 6 :=
sorry

end min_abs_phi_l661_66142


namespace total_books_l661_66124

theorem total_books (tim_books mike_books : ℕ) 
  (h1 : tim_books = 22) 
  (h2 : mike_books = 20) : 
  tim_books + mike_books = 42 := by
sorry

end total_books_l661_66124


namespace guam_stay_duration_l661_66132

/-- Calculates the number of days spent in Guam given the regular plan cost, international data cost per day, and total charges for the month. -/
def days_in_guam (regular_plan : ℚ) (intl_data_cost : ℚ) (total_charges : ℚ) : ℚ :=
  (total_charges - regular_plan) / intl_data_cost

/-- Theorem stating that given the specific costs in the problem, the number of days in Guam is 10. -/
theorem guam_stay_duration :
  let regular_plan : ℚ := 175
  let intl_data_cost : ℚ := 3.5
  let total_charges : ℚ := 210
  days_in_guam regular_plan intl_data_cost total_charges = 10 := by
  sorry

end guam_stay_duration_l661_66132


namespace integer_between_sqrt_11_and_sqrt_19_l661_66125

theorem integer_between_sqrt_11_and_sqrt_19 :
  ∃! x : ℤ, Real.sqrt 11 < x ∧ x < Real.sqrt 19 :=
by
  -- The proof goes here
  sorry

end integer_between_sqrt_11_and_sqrt_19_l661_66125


namespace triangle_side_length_l661_66196

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = π / 4 →
  Real.sin A + Real.sin (B - C) = 2 * Real.sqrt 2 * Real.sin (2 * C) →
  (1 / 2) * b * c * Real.sin A = 1 →
  a = Real.sqrt 5 :=
sorry

end triangle_side_length_l661_66196


namespace marnie_bracelets_l661_66101

/-- The number of bags of 50 beads Marnie bought -/
def bags_50 : ℕ := 5

/-- The number of bags of 100 beads Marnie bought -/
def bags_100 : ℕ := 2

/-- The number of beads in each bag of 50 -/
def beads_per_bag_50 : ℕ := 50

/-- The number of beads in each bag of 100 -/
def beads_per_bag_100 : ℕ := 100

/-- The number of beads needed to make one bracelet -/
def beads_per_bracelet : ℕ := 50

/-- The total number of beads Marnie bought -/
def total_beads : ℕ := bags_50 * beads_per_bag_50 + bags_100 * beads_per_bag_100

/-- The number of bracelets Marnie can make -/
def bracelets : ℕ := total_beads / beads_per_bracelet

theorem marnie_bracelets : bracelets = 9 := by sorry

end marnie_bracelets_l661_66101


namespace eighth_of_two_to_forty_l661_66110

theorem eighth_of_two_to_forty (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ x → x = 37 := by
  sorry

end eighth_of_two_to_forty_l661_66110


namespace stating_equal_probability_for_all_methods_l661_66103

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- The total number of components -/
def total_components : ℕ := 100

/-- The number of items to be sampled -/
def sample_size : ℕ := 20

/-- The number of first-grade items -/
def first_grade : ℕ := 20

/-- The number of second-grade items -/
def second_grade : ℕ := 30

/-- The number of third-grade items -/
def third_grade : ℕ := 50

/-- The probability of selecting any individual component -/
def selection_probability : ℚ := 1 / 5

/-- 
  Theorem stating that for all sampling methods, 
  the probability of selecting any individual component is 1/5
-/
theorem equal_probability_for_all_methods (method : SamplingMethod) : 
  (selection_probability : ℚ) = 1 / 5 := by sorry

end stating_equal_probability_for_all_methods_l661_66103


namespace similar_triangles_leg_sum_l661_66129

theorem similar_triangles_leg_sum (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a^2 + b^2 = c^2 →
  d^2 + e^2 = f^2 →
  (1/2) * a * b = 24 →
  (1/2) * d * e = 600 →
  c = 13 →
  (a / d)^2 = (b / e)^2 →
  d + e = 85 := by
sorry

end similar_triangles_leg_sum_l661_66129


namespace min_value_x_minus_3y_l661_66185

theorem min_value_x_minus_3y (x y : ℝ) (hx : x > 1) (hy : y < 0) (h : 3 * y * (1 - x) = x + 8) :
  ∀ z, x - 3 * y ≥ z → z ≤ 8 :=
sorry

end min_value_x_minus_3y_l661_66185


namespace negation_of_proposition_negation_of_inequality_l661_66164

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end negation_of_proposition_negation_of_inequality_l661_66164


namespace root_quadratic_equation_l661_66158

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
sorry

end root_quadratic_equation_l661_66158


namespace fewest_students_twenty_two_satisfies_fewest_students_is_22_l661_66151

theorem fewest_students (n : ℕ) : (n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6) → n ≥ 22 :=
by sorry

theorem twenty_two_satisfies : 22 % 5 = 2 ∧ 22 % 6 = 4 ∧ 22 % 8 = 6 :=
by sorry

theorem fewest_students_is_22 : 
  ∃ n : ℕ, n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ ∀ m : ℕ, (m % 5 = 2 ∧ m % 6 = 4 ∧ m % 8 = 6) → m ≥ n :=
by sorry

end fewest_students_twenty_two_satisfies_fewest_students_is_22_l661_66151


namespace total_present_age_is_72_l661_66198

/-- Given three people p, q, and r, prove that their total present age is 72 years -/
theorem total_present_age_is_72 
  (p q r : ℕ) -- Present ages of p, q, and r
  (h1 : p - 12 = (q - 12) / 2) -- 12 years ago, p was half of q's age
  (h2 : r - 12 = (p - 12) + (q - 12) - 3) -- r was 3 years younger than the sum of p and q's ages 12 years ago
  (h3 : ∃ (x : ℕ), p = 3*x ∧ q = 4*x ∧ r = 5*x) -- The ratio of their present ages is 3 : 4 : 5
  : p + q + r = 72 := by
  sorry


end total_present_age_is_72_l661_66198


namespace edward_lawn_mowing_earnings_l661_66190

/-- Edward's lawn mowing earnings problem -/
theorem edward_lawn_mowing_earnings 
  (rate : ℕ) -- Rate per lawn mowed
  (total_lawns : ℕ) -- Total number of lawns to mow
  (forgotten_lawns : ℕ) -- Number of lawns forgotten
  (h1 : rate = 4) -- Edward earns 4 dollars for each lawn
  (h2 : total_lawns = 17) -- Edward had 17 lawns to mow
  (h3 : forgotten_lawns = 9) -- Edward forgot to mow 9 lawns
  : (total_lawns - forgotten_lawns) * rate = 32 := by
  sorry

end edward_lawn_mowing_earnings_l661_66190


namespace die_visible_combinations_l661_66152

/-- A die is represented as a cube with 6 faces, 12 edges, and 8 vertices -/
structure Die :=
  (faces : Fin 6)
  (edges : Fin 12)
  (vertices : Fin 8)

/-- The number of visible faces from a point in space can be 1, 2, or 3 -/
inductive VisibleFaces
  | one
  | two
  | three

/-- The number of combinations for each type of view -/
def combinationsForView (v : VisibleFaces) : ℕ :=
  match v with
  | VisibleFaces.one => 6    -- One face visible: 6 possibilities
  | VisibleFaces.two => 12   -- Two faces visible: 12 possibilities
  | VisibleFaces.three => 8  -- Three faces visible: 8 possibilities

/-- The total number of different visible face combinations -/
def totalCombinations (d : Die) : ℕ :=
  (combinationsForView VisibleFaces.one) +
  (combinationsForView VisibleFaces.two) +
  (combinationsForView VisibleFaces.three)

theorem die_visible_combinations (d : Die) :
  totalCombinations d = 26 := by
  sorry

end die_visible_combinations_l661_66152


namespace orthogonal_families_l661_66199

/-- A family of curves in the x-y plane -/
structure Curve :=
  (equation : ℝ → ℝ → ℝ → Prop)

/-- The given family of curves x^2 + y^2 = 2ax -/
def given_family : Curve :=
  ⟨λ a x y ↦ x^2 + y^2 = 2*a*x⟩

/-- The orthogonal family of curves x^2 + y^2 = Cy -/
def orthogonal_family : Curve :=
  ⟨λ C x y ↦ x^2 + y^2 = C*y⟩

/-- Two curves are orthogonal if their tangent lines are perpendicular at each intersection point -/
def orthogonal (c1 c2 : Curve) : Prop :=
  ∀ a C x y, c1.equation a x y → c2.equation C x y →
    ∃ m1 m2 : ℝ, (m1 * m2 = -1) ∧
      (∀ h, h ≠ 0 → (c1.equation a (x + h) (y + m1*h) ↔ c1.equation a x y)) ∧
      (∀ h, h ≠ 0 → (c2.equation C (x + h) (y + m2*h) ↔ c2.equation C x y))

/-- The main theorem stating that the given family and the orthogonal family are indeed orthogonal -/
theorem orthogonal_families : orthogonal given_family orthogonal_family :=
sorry

end orthogonal_families_l661_66199


namespace rectangle_area_is_48_l661_66109

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle PQRS with points U and V on its diagonal -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point
  U : Point
  V : Point

/-- Given conditions for the rectangle problem -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- PQRS is a rectangle (implied by other conditions)
  -- PQ is parallel to RS (implied by rectangle property)
  (rect.P.x - rect.Q.x = rect.R.x - rect.S.x) ∧ 
  (rect.P.y - rect.Q.y = rect.R.y - rect.S.y) ∧ 
  -- PQ = RS
  ((rect.P.x - rect.Q.x)^2 + (rect.P.y - rect.Q.y)^2 = 
   (rect.R.x - rect.S.x)^2 + (rect.R.y - rect.S.y)^2) ∧
  -- U and V lie on diagonal PS
  ((rect.U.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.U.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  ((rect.V.x - rect.P.x) * (rect.S.y - rect.P.y) = 
   (rect.V.y - rect.P.y) * (rect.S.x - rect.P.x)) ∧
  -- U is between P and V
  ((rect.U.x - rect.P.x) * (rect.V.x - rect.U.x) ≥ 0) ∧
  ((rect.U.y - rect.P.y) * (rect.V.y - rect.U.y) ≥ 0) ∧
  -- Angle PUV = 90°
  ((rect.P.x - rect.U.x) * (rect.V.x - rect.U.x) + 
   (rect.P.y - rect.U.y) * (rect.V.y - rect.U.y) = 0) ∧
  -- Angle QVR = 90°
  ((rect.Q.x - rect.V.x) * (rect.R.x - rect.V.x) + 
   (rect.Q.y - rect.V.y) * (rect.R.y - rect.V.y) = 0) ∧
  -- PU = 4
  ((rect.P.x - rect.U.x)^2 + (rect.P.y - rect.U.y)^2 = 16) ∧
  -- UV = 2
  ((rect.U.x - rect.V.x)^2 + (rect.U.y - rect.V.y)^2 = 4) ∧
  -- VS = 6
  ((rect.V.x - rect.S.x)^2 + (rect.V.y - rect.S.y)^2 = 36)

/-- The area of a rectangle -/
def rectangle_area (rect : Rectangle) : ℝ :=
  abs ((rect.P.x - rect.Q.x) * (rect.Q.y - rect.R.y) - 
       (rect.P.y - rect.Q.y) * (rect.Q.x - rect.R.x))

/-- Theorem stating that the area of the rectangle is 48 -/
theorem rectangle_area_is_48 (rect : Rectangle) : 
  rectangle_conditions rect → rectangle_area rect = 48 := by
  sorry

end rectangle_area_is_48_l661_66109


namespace quadratic_always_positive_l661_66186

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions for a valid triangle
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the quadratic expression
def quadratic_expr (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem quadratic_always_positive (t : Triangle) :
  ∀ x : ℝ, quadratic_expr t x > 0 := by
  sorry

end quadratic_always_positive_l661_66186


namespace poppy_seed_count_l661_66115

def total_slices : ℕ := 58

theorem poppy_seed_count (x : ℕ) 
  (h1 : x ≤ total_slices)
  (h2 : Nat.choose x 3 = Nat.choose (total_slices - x) 2 * x) :
  total_slices - x = 21 := by
  sorry

end poppy_seed_count_l661_66115


namespace inequality_solution_set_l661_66100

theorem inequality_solution_set :
  {x : ℝ | (1/2: ℝ)^x ≤ (1/2 : ℝ)^(x+1) + 1} = {x : ℝ | x ≥ -1} := by
  sorry

end inequality_solution_set_l661_66100


namespace nested_fraction_equality_l661_66178

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_equality_l661_66178


namespace product_expansion_l661_66128

theorem product_expansion (x y : ℝ) :
  (3 * x + 4) * (2 * x + 6 * y + 7) = 6 * x^2 + 18 * x * y + 29 * x + 24 * y + 28 := by
  sorry

end product_expansion_l661_66128


namespace vaccine_waiting_time_l661_66163

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

end vaccine_waiting_time_l661_66163


namespace last_triangle_perimeter_l661_66191

/-- Represents a triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence based on the incircle tangent points -/
def nextTriangle (t : Triangle) : Triangle :=
  sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The sequence of triangles starting from the initial triangle -/
def triangleSequence : ℕ → Triangle
  | 0 => { a := 1015, b := 1016, c := 1017 }
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ :=
  sorry

theorem last_triangle_perimeter :
  perimeter (triangleSequence lastValidTriangleIndex) = 762 / 128 :=
sorry

end last_triangle_perimeter_l661_66191


namespace statements_b_and_c_correct_l661_66122

theorem statements_b_and_c_correct :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end statements_b_and_c_correct_l661_66122


namespace simplify_radical_expression_l661_66139

theorem simplify_radical_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 20 + Real.sqrt 500 / Real.sqrt 5 + 2 * Real.sqrt 45 = 4 * Real.sqrt 5 + 10 := by
  sorry

end simplify_radical_expression_l661_66139


namespace sweets_distribution_l661_66121

theorem sweets_distribution (total : ℕ) (june_sweets : ℕ) : 
  total = 90 →
  (3 : ℕ) * june_sweets = 4 * ((2 : ℕ) * june_sweets / 3) → 
  (1 : ℕ) * june_sweets / 2 + (3 : ℕ) * june_sweets / 4 + june_sweets = total →
  june_sweets = 40 := by
sorry

end sweets_distribution_l661_66121


namespace expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l661_66117

-- Problem 1
theorem expand_expression_1 (x y : ℝ) :
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
sorry

-- Problem 2
theorem expand_expression_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
sorry

-- Problem 3
theorem expand_expression_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
sorry

-- Problem 4
theorem simplified_calculation :
  2010^2 - 2011 * 2009 = 1 :=
sorry

end expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l661_66117


namespace divisibility_proof_l661_66104

theorem divisibility_proof (n : ℕ) : 
  n = 6268440 → n % 8 = 0 ∧ n % 66570 = 0 := by
  sorry

end divisibility_proof_l661_66104


namespace abc_sum_mod_five_l661_66141

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧
  0 < b ∧ b < 5 ∧
  0 < c ∧ c < 5 ∧
  (a * b * c) % 5 = 1 ∧
  (4 * c) % 5 = 3 ∧
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end abc_sum_mod_five_l661_66141


namespace not_power_of_two_l661_66192

theorem not_power_of_two (a b : ℕ+) : ¬ ∃ k : ℕ, (36 * a + b) * (a + 36 * b) = 2^k := by
  sorry

end not_power_of_two_l661_66192


namespace infinite_divisible_sequence_l661_66173

theorem infinite_divisible_sequence : 
  ∃ (f : ℕ → ℕ), 
    (∀ k, f k > 0) ∧ 
    (∀ k, k < k.succ → f k < f k.succ) ∧ 
    (∀ k, (2 ^ (f k) + 3 ^ (f k)) % (f k)^2 = 0) :=
sorry

end infinite_divisible_sequence_l661_66173


namespace units_digit_of_product_l661_66149

theorem units_digit_of_product (a b c : ℕ) : 
  (4^503 * 3^401 * 15^402) % 10 = 0 := by
  sorry

end units_digit_of_product_l661_66149


namespace sin_sum_identity_l661_66193

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end sin_sum_identity_l661_66193


namespace cubic_polynomial_root_sum_l661_66145

theorem cubic_polynomial_root_sum (f : ℝ → ℝ) (r₁ r₂ r₃ : ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  ((f (1/2) + f (-1/2)) / f 0 = 1003) →
  (1 / (r₁ * r₂) + 1 / (r₂ * r₃) + 1 / (r₃ * r₁) = 2002) :=
by sorry

end cubic_polynomial_root_sum_l661_66145


namespace log2_odd_and_increasing_l661_66114

open Real

-- Define the function f(x) = log₂ x
noncomputable def f (x : ℝ) : ℝ := log x / log 2

-- Theorem statement
theorem log2_odd_and_increasing :
  (∀ x > 0, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end log2_odd_and_increasing_l661_66114


namespace intersection_points_range_l661_66172

/-- The curve equation x^2 + (y+3)^2 = 4 -/
def curve (x y : ℝ) : Prop := x^2 + (y+3)^2 = 4

/-- The line equation y = k(x-2) -/
def line (k x y : ℝ) : Prop := y = k*(x-2)

/-- The theorem stating the range of k for which the curve and line have two distinct intersection points -/
theorem intersection_points_range :
  ∀ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ y₁ ∧ curve x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧ 
    y₁ ≥ -3 ∧ y₂ ≥ -3) ↔ 
  (5/12 < k ∧ k ≤ 3/4) :=
sorry

end intersection_points_range_l661_66172


namespace sphere_cylinder_equal_area_l661_66126

-- Define the constants for the cylinder
def cylinder_height : ℝ := 10
def cylinder_diameter : ℝ := 10

-- Define the theorem
theorem sphere_cylinder_equal_area (r : ℝ) :
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 5 := by
  sorry


end sphere_cylinder_equal_area_l661_66126


namespace division_subtraction_problem_l661_66176

theorem division_subtraction_problem : (12 / (2/3)) - 4 = 14 := by
  sorry

end division_subtraction_problem_l661_66176


namespace stratified_sampling_theorem_l661_66127

/-- Represents the stratified sampling scenario -/
structure SamplingScenario where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  sample_size : ℕ

/-- The specific scenario from the problem -/
def track_team : SamplingScenario :=
  { total_members := 42
  , boys := 28
  , girls := 14
  , sample_size := 6 }

/-- The probability of an individual being selected -/
def selection_probability (s : SamplingScenario) : ℚ :=
  s.sample_size / s.total_members

/-- The number of boys selected in stratified sampling -/
def boys_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.boys) / s.total_members

/-- The number of girls selected in stratified sampling -/
def girls_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.girls) / s.total_members

theorem stratified_sampling_theorem (s : SamplingScenario) :
  s = track_team →
  selection_probability s = 1/7 ∧
  boys_selected s = 4 ∧
  girls_selected s = 2 := by
  sorry

end stratified_sampling_theorem_l661_66127


namespace hyperbola_eccentricity_l661_66156

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 
    and asymptotes y = ±(√3/2)x is √7/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = Real.sqrt 3 / 2) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 7 / 2 := by
  sorry

end hyperbola_eccentricity_l661_66156


namespace reflection_theorem_l661_66148

def P : ℝ × ℝ := (1, 2)

-- Reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Reflection across origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem reflection_theorem :
  reflect_x P = (1, -2) ∧ reflect_origin P = (-1, -2) := by
  sorry

end reflection_theorem_l661_66148


namespace distribute_five_two_correct_l661_66183

def number_of_correct_locations : ℕ := 2
def total_objects : ℕ := 5

/-- The number of ways to distribute n distinct objects to n distinct locations
    such that exactly k objects are in their correct locations -/
def distribute (n k : ℕ) : ℕ := sorry

theorem distribute_five_two_correct :
  distribute total_objects number_of_correct_locations = 20 := by sorry

end distribute_five_two_correct_l661_66183


namespace units_digit_of_special_number_l661_66147

def is_product_of_one_digit_numbers (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), (factors.all (λ x => x > 0 ∧ x < 10)) ∧ 
    (factors.prod = n)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem units_digit_of_special_number (n : ℕ) :
  n > 10 ∧ 
  is_product_of_one_digit_numbers n ∧ 
  Odd (digit_product n) →
  n % 10 = 5 := by
sorry

end units_digit_of_special_number_l661_66147


namespace complex_division_product_l661_66160

/-- Given (2+3i)/i = a+bi, where a and b are real numbers and i is the imaginary unit, prove that ab = 6 -/
theorem complex_division_product (a b : ℝ) : (Complex.I : ℂ)⁻¹ * (2 + 3 * Complex.I) = a + b * Complex.I → a * b = 6 := by
  sorry

end complex_division_product_l661_66160


namespace cube_diff_even_iff_sum_even_l661_66137

theorem cube_diff_even_iff_sum_even (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) :=
by sorry

end cube_diff_even_iff_sum_even_l661_66137


namespace composition_three_reflections_is_glide_reflection_l661_66175

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for lines in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a reflection transformation
def Reflection (l : Line2D) : Point2D → Point2D := sorry

-- Define a translation transformation
def Translation (dx dy : ℝ) : Point2D → Point2D := sorry

-- Define a glide reflection transformation
def GlideReflection (l : Line2D) (t : ℝ) : Point2D → Point2D := sorry

-- Define a predicate to check if three lines pass through the same point
def passThroughSamePoint (l1 l2 l3 : Line2D) : Prop := sorry

-- Define a predicate to check if three lines are parallel to the same line
def parallelToSameLine (l1 l2 l3 : Line2D) : Prop := sorry

-- Theorem statement
theorem composition_three_reflections_is_glide_reflection 
  (l1 l2 l3 : Line2D) 
  (h1 : ¬ passThroughSamePoint l1 l2 l3) 
  (h2 : ¬ parallelToSameLine l1 l2 l3) :
  ∃ (l : Line2D) (t : ℝ), 
    ∀ p : Point2D, 
      (Reflection l3 ∘ Reflection l2 ∘ Reflection l1) p = GlideReflection l t p :=
sorry

end composition_three_reflections_is_glide_reflection_l661_66175


namespace marketing_specialization_percentage_l661_66167

theorem marketing_specialization_percentage
  (initial_finance : Real)
  (increased_finance : Real)
  (marketing_after_increase : Real)
  (h1 : initial_finance = 88)
  (h2 : increased_finance = 90)
  (h3 : marketing_after_increase = 43.333333333333336)
  (h4 : increased_finance - initial_finance = 2) :
  initial_finance + marketing_after_increase + 2 = 45.333333333333336 + 88 := by
  sorry

end marketing_specialization_percentage_l661_66167


namespace cubic_function_theorem_l661_66111

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3

theorem cubic_function_theorem (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0) →  -- two distinct zeros
  (∃ x₀ : ℝ, f c x₀ = 0 ∧ ∀ x : ℝ, f c x ≤ f c x₀) →  -- one zero is the maximum point
  c = -2 :=
sorry

end cubic_function_theorem_l661_66111


namespace sector_area_l661_66112

/-- The area of a circular sector with central angle 5π/7 and perimeter 5π+14 is 35π/2 -/
theorem sector_area (r : ℝ) (h1 : r > 0) : 
  (5 / 7 * π * r + 2 * r = 5 * π + 14) →
  (1 / 2 * (5 / 7 * π) * r^2 = 35 * π / 2) := by
sorry


end sector_area_l661_66112


namespace buyOneGetOneFreeIsCheaper_finalCostIs216_l661_66107

/-- Represents the total cost of Pauline's purchase with a given discount and sales tax. -/
def totalCost (totalBeforeTax : ℝ) (selectedItemsTotal : ℝ) (discount : ℝ) (salesTaxRate : ℝ) : ℝ :=
  let discountedTotal := totalBeforeTax - selectedItemsTotal * discount
  discountedTotal * (1 + salesTaxRate)

/-- Theorem stating that the Buy One, Get One Free offer is cheaper than the 15% discount offer. -/
theorem buyOneGetOneFreeIsCheaper :
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let remainingItemsTotal : ℝ := totalBeforeTax - selectedItemsTotal
  let discountRate : ℝ := 0.15
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate <
  totalCost totalBeforeTax selectedItemsTotal discountRate salesTaxRate :=
by sorry

/-- Calculates the final cost with the Buy One, Get One Free offer. -/
def finalCost : ℝ :=
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate

/-- Theorem stating that the final cost is $216. -/
theorem finalCostIs216 : finalCost = 216 :=
by sorry

end buyOneGetOneFreeIsCheaper_finalCostIs216_l661_66107


namespace jewelry_sales_problem_l661_66169

/-- Represents the jewelry sales problem --/
theorem jewelry_sales_problem 
  (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (bracelets_sold earrings_sold ensembles_sold : ℕ)
  (total_revenue : ℚ)
  (h1 : necklace_price = 25)
  (h2 : bracelet_price = 15)
  (h3 : earring_price = 10)
  (h4 : ensemble_price = 45)
  (h5 : bracelets_sold = 10)
  (h6 : earrings_sold = 20)
  (h7 : ensembles_sold = 2)
  (h8 : total_revenue = 565) :
  ∃ (necklaces_sold : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earring_price * earrings_sold + 
    ensemble_price * ensembles_sold = total_revenue ∧
    necklaces_sold = 5 := by
  sorry

end jewelry_sales_problem_l661_66169


namespace cube_root_five_to_seven_sum_l661_66138

theorem cube_root_five_to_seven_sum : 
  (5^7 + 5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 25 * (5^2)^(1/3) := by
  sorry

end cube_root_five_to_seven_sum_l661_66138
