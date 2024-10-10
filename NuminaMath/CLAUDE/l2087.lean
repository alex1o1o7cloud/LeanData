import Mathlib

namespace money_distribution_inconsistency_l2087_208702

/-- Represents the money distribution problem with aunts and children --/
theorem money_distribution_inconsistency 
  (jade_money : ℕ) 
  (julia_money : ℕ) 
  (jack_money : ℕ) 
  (john_money : ℕ) 
  (jane_money : ℕ) 
  (total_after : ℕ) 
  (aunt_mary_gift : ℕ) 
  (aunt_susan_gift : ℕ) 
  (h1 : jade_money = 38)
  (h2 : julia_money = jade_money / 2)
  (h3 : jack_money = 12)
  (h4 : john_money = 15)
  (h5 : jane_money = 20)
  (h6 : total_after = 225)
  (h7 : aunt_mary_gift = 65)
  (h8 : aunt_susan_gift = 70) : 
  ¬(∃ (aunt_lucy_gift : ℕ) (individual_gift : ℕ),
    jade_money + julia_money + jack_money + john_money + jane_money + 
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = total_after ∧
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = 5 * individual_gift) :=
sorry


end money_distribution_inconsistency_l2087_208702


namespace max_primitive_dinosaur_cells_l2087_208788

/-- Represents a dinosaur as a tree -/
structure Dinosaur where
  cells : ℕ
  is_connected : Bool
  max_degree : ℕ

/-- Defines a primitive dinosaur -/
def is_primitive (d : Dinosaur) : Prop :=
  ∀ (d1 d2 : Dinosaur), d.cells ≠ d1.cells + d2.cells ∨ d1.cells < 2007 ∨ d2.cells < 2007

/-- The main theorem stating the maximum number of cells in a primitive dinosaur -/
theorem max_primitive_dinosaur_cells :
  ∀ (d : Dinosaur),
    d.cells ≥ 2007 →
    d.is_connected = true →
    d.max_degree = 4 →
    is_primitive d →
    d.cells ≤ 8025 :=
sorry

end max_primitive_dinosaur_cells_l2087_208788


namespace min_value_theorem_l2087_208748

/-- The hyperbola equation -/
def hyperbola (m n x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The condition that the hyperbola and ellipse have the same foci -/
def same_foci (m n : ℝ) : Prop := m + n = 1

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola : ∃ x y, hyperbola m n x y)
  (h_ellipse : ∃ x y, ellipse x y)
  (h_foci : same_foci m n) :
  (∀ m' n', m' > 0 → n' > 0 → same_foci m' n' → 4/m + 1/n ≤ 4/m' + 1/n') ∧ 
  (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ same_foci m₀ n₀ ∧ 4/m₀ + 1/n₀ = 9) :=
sorry

end min_value_theorem_l2087_208748


namespace chocolate_percentage_proof_l2087_208706

/-- Represents the number of each type of chocolate bar -/
def chocolate_count : ℕ := 25

/-- Represents the number of different types of chocolate bars -/
def chocolate_types : ℕ := 4

/-- Calculates the total number of chocolate bars -/
def total_chocolates : ℕ := chocolate_count * chocolate_types

/-- Represents the percentage as a rational number -/
def percentage_per_type : ℚ := chocolate_count / total_chocolates

theorem chocolate_percentage_proof :
  percentage_per_type = 1 / 4 := by sorry

end chocolate_percentage_proof_l2087_208706


namespace triangle_area_l2087_208725

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 + c^2 = a^2 + b*c →
  b * c * Real.cos A = 4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
sorry

end triangle_area_l2087_208725


namespace complex_power_sum_l2087_208794

theorem complex_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 5)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 10) :
  α₁^6 + α₂^6 + α₃^6 = 44 := by sorry

end complex_power_sum_l2087_208794


namespace sequence_properties_l2087_208703

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_properties (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 4 = 24 →
  b 1 = 0 →
  (∀ n : ℕ, b n + b (n + 1) = a n) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^(n - 1) + (-1)^n) := by
  sorry

end sequence_properties_l2087_208703


namespace repeating_decimal_as_fraction_l2087_208739

def repeating_decimal : ℚ := 0.157142857142857

theorem repeating_decimal_as_fraction :
  repeating_decimal = 10690 / 68027 := by sorry

end repeating_decimal_as_fraction_l2087_208739


namespace cube_circumscribed_sphere_radius_l2087_208745

/-- The radius of the circumscribed sphere of a cube with edge length 1 is √3/2 -/
theorem cube_circumscribed_sphere_radius :
  let cube_edge_length : ℝ := 1
  let circumscribed_sphere_radius : ℝ := (Real.sqrt 3) / 2
  cube_edge_length = 1 →
  circumscribed_sphere_radius = (Real.sqrt 3) / 2 :=
by
  sorry


end cube_circumscribed_sphere_radius_l2087_208745


namespace complement_intersection_subset_range_l2087_208787

-- Define the sets A and B
def A : Set ℝ := {x | 2 < x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Theorem 1: Prove the complement intersection when a = 2
theorem complement_intersection :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem 2: Prove the range of a for which B is a subset of A
theorem subset_range :
  {a : ℝ | B a ⊆ A} = {a | 3 ≤ a ∧ a ≤ 4} := by sorry

end complement_intersection_subset_range_l2087_208787


namespace exponent_division_l2087_208743

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^8 / a^2 = a^6 := by
  sorry

end exponent_division_l2087_208743


namespace largest_number_l2087_208797

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 997/1000) 
  (hb : b = 9799/10000) 
  (hc : c = 999/1000) 
  (hd : d = 9979/10000) 
  (he : e = 979/1000) : 
  c = max a (max b (max c (max d e))) := by
  sorry

end largest_number_l2087_208797


namespace number_properties_l2087_208769

def number : ℕ := 52300600

-- Define a function to get the digit at a specific position
def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / (10 ^ (pos - 1))) % 10

-- Define a function to get the value represented by a digit at a specific position
def value_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (digit_at_position n pos) * (10 ^ (pos - 1))

-- Define a function to convert a number to its word representation
def number_to_words (n : ℕ) : String :=
  sorry -- Implementation details omitted

theorem number_properties :
  (digit_at_position number 8 = 2) ∧
  (value_at_position number 8 = 20000000) ∧
  (digit_at_position number 9 = 5) ∧
  (value_at_position number 9 = 500000000) ∧
  (number_to_words number = "five hundred twenty-three million six hundred") := by
  sorry

end number_properties_l2087_208769


namespace chatterbox_jokes_l2087_208704

def n : ℕ := 10  -- number of chatterboxes

-- Sum of natural numbers from 1 to m
def sum_to(m : ℕ) : ℕ := m * (m + 1) / 2

-- Total number of jokes told
def total_jokes : ℕ := sum_to 100 + sum_to 99

theorem chatterbox_jokes :
  total_jokes / n = 1000 :=
sorry

end chatterbox_jokes_l2087_208704


namespace triangle_hyperbola_ratio_l2087_208720

-- Define the right triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  (xB - xA)^2 + (yB - yA)^2 = 3^2 ∧
  (xC - xA)^2 + (yC - yA)^2 = 1^2 ∧
  (xB - xA) * (xC - xA) + (yB - yA) * (yC - yA) = 0

-- Define the hyperbola passing through A and intersecting AB at D
def Hyperbola (A B C D : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  let (xD, yD) := D
  a > 0 ∧ b > 0 ∧
  xA^2 / a^2 - yA^2 / b^2 = 1 ∧
  xD^2 / a^2 - yD^2 / b^2 = 1 ∧
  (xD - xA) * (yB - yA) = (yD - yA) * (xB - xA)

-- Theorem statement
theorem triangle_hyperbola_ratio 
  (A B C D : ℝ × ℝ) (a b : ℝ) :
  Triangle A B C → Hyperbola A B C D a b →
  let (xA, yA) := A
  let (xB, yB) := B
  let (xD, yD) := D
  Real.sqrt ((xD - xA)^2 + (yD - yA)^2) / Real.sqrt ((xB - xD)^2 + (yB - yD)^2) = 4 :=
sorry

end triangle_hyperbola_ratio_l2087_208720


namespace arithmetic_progression_probability_l2087_208718

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when tossing three dice -/
def total_outcomes : ℕ := num_faces ^ 3

/-- A function that checks if three numbers form an arithmetic progression with common difference 2 -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 2 ∧ c = b + 2) ∨ (b = a - 2 ∧ c = b - 2) ∨
  (a = b + 2 ∧ c = a + 2) ∨ (c = b + 2 ∧ a = c + 2) ∨
  (a = b - 2 ∧ c = a - 2) ∨ (c = b - 2 ∧ a = c - 2)

/-- The number of favorable outcomes (i.e., outcomes that form an arithmetic progression) -/
def favorable_outcomes : ℕ := 12

/-- The theorem stating the probability of getting an arithmetic progression -/
theorem arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 :=
sorry

end arithmetic_progression_probability_l2087_208718


namespace abc_fraction_value_l2087_208744

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 9) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 := by
  sorry

end abc_fraction_value_l2087_208744


namespace windows_preference_count_l2087_208754

theorem windows_preference_count (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac_pref = 60 → 
  no_pref = 90 → 
  ∃ (windows_pref : ℕ), 
    windows_pref = total - (mac_pref + (mac_pref / 3) + no_pref) ∧ 
    windows_pref = 40 := by
  sorry

end windows_preference_count_l2087_208754


namespace supermarket_profit_l2087_208777

/-- Represents the daily sales quantity as a function of the selling price. -/
def sales_quantity (x : ℤ) : ℤ := -5 * x + 150

/-- Represents the daily profit as a function of the selling price. -/
def daily_profit (x : ℤ) : ℤ := (x - 8) * (sales_quantity x)

theorem supermarket_profit (x : ℤ) (h1 : 8 ≤ x) (h2 : x ≤ 15) :
  (daily_profit 14 = 480) ∧
  (∀ y : ℤ, 8 ≤ y → y ≤ 15 → daily_profit y ≤ daily_profit 15) ∧
  (daily_profit 15 = 525) :=
sorry


end supermarket_profit_l2087_208777


namespace smallest_b_for_factorization_l2087_208751

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 2008 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 2008 = (x + r) * (x + s)) ∧
  b = 259 :=
by sorry

end smallest_b_for_factorization_l2087_208751


namespace function_satisfying_inequality_is_constant_l2087_208786

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (f x - f y)^2 ≤ |x - y|^3) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end function_satisfying_inequality_is_constant_l2087_208786


namespace eBook_readers_count_l2087_208737

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem eBook_readers_count : total_readers = 82 := by
  sorry

end eBook_readers_count_l2087_208737


namespace binomial_coefficient_1450_2_l2087_208761

theorem binomial_coefficient_1450_2 : Nat.choose 1450 2 = 1050205 := by
  sorry

end binomial_coefficient_1450_2_l2087_208761


namespace N_mod_45_l2087_208708

/-- N is the number formed by concatenating integers from 1 to 52 -/
def N : ℕ := sorry

theorem N_mod_45 : N % 45 = 37 := by sorry

end N_mod_45_l2087_208708


namespace sixth_term_term_1994_l2087_208747

-- Define the sequence
def a (n : ℕ) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end sixth_term_term_1994_l2087_208747


namespace cube_probability_l2087_208756

/-- A cube with side length 3 -/
def Cube := Fin 3 → Fin 3 → Fin 3

/-- The number of unit cubes in the larger cube -/
def totalCubes : ℕ := 27

/-- The number of unit cubes with exactly two painted faces -/
def twoPaintedFaces : ℕ := 4

/-- The number of unit cubes with no painted faces -/
def noPaintedFaces : ℕ := 8

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def probability : ℚ := 32 / 351

theorem cube_probability : 
  probability = (twoPaintedFaces * noPaintedFaces : ℚ) / (totalCubes.choose 2) := by sorry

end cube_probability_l2087_208756


namespace tangent_line_determines_b_l2087_208772

/-- A curve defined by y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = mx + c -/
def line (m c : ℝ) (x : ℝ) : ℝ := m*x + c

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 ∧
  curve_derivative a 1 = 2 →
  b = 3 := by sorry

end tangent_line_determines_b_l2087_208772


namespace unique_function_property_l2087_208775

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 0 ↔ x = 0)
  (h2 : ∀ x y, f (x^2 + y * f x) + f (y^2 + x * f y) = (f (x + y))^2) :
  ∀ x, f x = x := by
  sorry

end unique_function_property_l2087_208775


namespace maintenance_check_increase_l2087_208731

theorem maintenance_check_increase (original_time new_time : ℝ) 
  (h1 : original_time = 50)
  (h2 : new_time = 60) :
  (new_time - original_time) / original_time * 100 = 20 := by
  sorry

end maintenance_check_increase_l2087_208731


namespace contrapositive_equivalence_l2087_208790

theorem contrapositive_equivalence (a b : ℝ) :
  (¬ (-Real.sqrt b < a ∧ a < Real.sqrt b) → ¬ (a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end contrapositive_equivalence_l2087_208790


namespace revenue_is_288_l2087_208779

/-- Represents the rental business with canoes and kayaks -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for a day given the rental business conditions -/
def calculate_revenue (business : RentalBusiness) : ℕ :=
  let kayaks := business.canoe_kayak_difference * 2
  let canoes := kayaks + business.canoe_kayak_difference
  kayaks * business.kayak_price + canoes * business.canoe_price

/-- Theorem stating that the total revenue for the day is $288 -/
theorem revenue_is_288 (business : RentalBusiness) 
    (h1 : business.canoe_price = 14)
    (h2 : business.kayak_price = 15)
    (h3 : business.canoe_kayak_ratio = 3 / 2)
    (h4 : business.canoe_kayak_difference = 4) :
  calculate_revenue business = 288 := by
  sorry

#eval calculate_revenue { 
  canoe_price := 14, 
  kayak_price := 15, 
  canoe_kayak_ratio := 3 / 2, 
  canoe_kayak_difference := 4 
}

end revenue_is_288_l2087_208779


namespace function_property_Z_function_property_Q_l2087_208719

-- For integers
theorem function_property_Z (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℤ, f x = 2 * x ∨ f x = 0) :=
sorry

-- For rationals (bonus)
theorem function_property_Q (f : ℚ → ℚ) :
  (∀ a b : ℚ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℚ, f x = 2 * x ∨ f x = 0) :=
sorry

end function_property_Z_function_property_Q_l2087_208719


namespace geometric_sequence_quadratic_one_root_l2087_208781

/-- If real numbers a, b, c form a geometric sequence, then the function f(x) = ax^2 + 2bx + c has exactly one real root. -/
theorem geometric_sequence_quadratic_one_root
  (a b c : ℝ) (h_geometric : b^2 = a*c) :
  ∃! x, a*x^2 + 2*b*x + c = 0 :=
by sorry

end geometric_sequence_quadratic_one_root_l2087_208781


namespace shopkeeper_profit_percentage_l2087_208710

def selling_price : ℝ := 1110
def cost_price : ℝ := 925

theorem shopkeeper_profit_percentage :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end shopkeeper_profit_percentage_l2087_208710


namespace width_to_perimeter_ratio_l2087_208767

/-- Represents a rectangular classroom -/
structure Classroom where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a classroom -/
def perimeter (c : Classroom) : ℝ := 2 * (c.length + c.width)

/-- Theorem: The ratio of width to perimeter for a 15x10 classroom is 1:5 -/
theorem width_to_perimeter_ratio (c : Classroom) 
  (h1 : c.length = 15) 
  (h2 : c.width = 10) : 
  c.width / perimeter c = 1 / 5 := by
  sorry

#check width_to_perimeter_ratio

end width_to_perimeter_ratio_l2087_208767


namespace max_value_of_f_in_interval_l2087_208782

def f (x : ℝ) : ℝ := x^2 + 3*x + 2

theorem max_value_of_f_in_interval :
  ∃ (M : ℝ), M = 42 ∧ 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧
  (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x = M) := by
  sorry

end max_value_of_f_in_interval_l2087_208782


namespace variance_scaled_and_shifted_l2087_208795

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (y : Fin n → ℝ) : ℝ := sorry

theorem variance_scaled_and_shifted
  (h : variance x = 1) :
  variance (fun i => 2 * x i + 1) = 4 := by sorry

end variance_scaled_and_shifted_l2087_208795


namespace math_club_teams_l2087_208734

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for each team -/
def girls_per_team : ℕ := 2

/-- The number of boys to be selected for each team -/
def boys_per_team : ℕ := 2

theorem math_club_teams : 
  (choose num_girls girls_per_team) * (choose num_boys boys_per_team) = 90 := by
  sorry

end math_club_teams_l2087_208734


namespace vector_c_determination_l2087_208778

/-- Given vectors a and b, if vector c satisfies the conditions, then c = (2, 1) -/
theorem vector_c_determination (a b c : ℝ × ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (1, 2)) 
  (hperp : (c.1 + b.1, c.2 + b.2) • a = 0)  -- (c + b) ⊥ a
  (hpar : ∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2))  -- (c - a) ∥ b
  : c = (2, 1) := by
  sorry

end vector_c_determination_l2087_208778


namespace lcm_gcd_product_l2087_208721

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = 2250 ∧ Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

#check lcm_gcd_product

end lcm_gcd_product_l2087_208721


namespace james_car_transaction_l2087_208798

/-- The amount James is out of pocket after selling his old car and buying a new one -/
def out_of_pocket (old_car_value : ℝ) (old_car_sale_percentage : ℝ) 
                  (new_car_sticker : ℝ) (new_car_buy_percentage : ℝ) : ℝ :=
  new_car_sticker * new_car_buy_percentage - old_car_value * old_car_sale_percentage

/-- Theorem stating that James is out of pocket by $11,000 -/
theorem james_car_transaction : 
  out_of_pocket 20000 0.8 30000 0.9 = 11000 := by
  sorry

end james_car_transaction_l2087_208798


namespace hairstylist_normal_haircut_price_l2087_208765

theorem hairstylist_normal_haircut_price :
  let normal_price : ℝ := x
  let special_price : ℝ := 6
  let trendy_price : ℝ := 8
  let normal_per_day : ℕ := 5
  let special_per_day : ℕ := 3
  let trendy_per_day : ℕ := 2
  let days_per_week : ℕ := 7
  let weekly_earnings : ℝ := 413
  (normal_price * (normal_per_day * days_per_week : ℝ) +
   special_price * (special_per_day * days_per_week : ℝ) +
   trendy_price * (trendy_per_day * days_per_week : ℝ) = weekly_earnings) →
  normal_price = 5 := by
sorry

end hairstylist_normal_haircut_price_l2087_208765


namespace mikes_current_age_l2087_208729

theorem mikes_current_age :
  ∀ (M B : ℕ),
  B = M / 2 →
  M - B = 24 - 16 →
  M = 16 :=
by
  sorry

end mikes_current_age_l2087_208729


namespace intersection_projection_distance_l2087_208753

/-- Given a line and a circle intersecting at two points, 
    prove that the distance between the projections of these points on the x-axis is 4. -/
theorem intersection_projection_distance (A B C D : ℝ × ℝ) : 
  -- Line equation
  (∀ (x y : ℝ), (x, y) ∈ {(x, y) | x - Real.sqrt 3 * y + 6 = 0} → 
    (A.1 - Real.sqrt 3 * A.2 + 6 = 0 ∧ B.1 - Real.sqrt 3 * B.2 + 6 = 0)) →
  -- Circle equation
  (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12) →
  -- A and B are distinct points
  A ≠ B →
  -- C and D are projections of A and B on x-axis
  (C = (A.1, 0) ∧ D = (B.1, 0)) →
  -- Distance between C and D is 4
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4 :=
by sorry


end intersection_projection_distance_l2087_208753


namespace quadratic_inequality_l2087_208709

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 2) * x + 2 < 0 ↔ 
    ((a < 0 ∧ (x < 2/a ∨ x > 1)) ∨
     (a = 0 ∧ x > 1) ∨
     (0 < a ∧ a < 2 ∧ 1 < x ∧ x < 2/a) ∨
     (a > 2 ∧ 2/a < x ∧ x < 1)) :=
by sorry

end quadratic_inequality_l2087_208709


namespace transportation_puzzle_l2087_208716

def is_valid_assignment (T R A N S P O B K : ℕ) : Prop :=
  T > R ∧ R > A ∧ A > N ∧ N < S ∧ S < P ∧ P < O ∧ O < R ∧ R < T ∧
  T > R ∧ R > O ∧ O < A ∧ A > B ∧ B < K ∧ K < A ∧
  T ≠ R ∧ T ≠ A ∧ T ≠ N ∧ T ≠ S ∧ T ≠ P ∧ T ≠ O ∧ T ≠ B ∧ T ≠ K ∧
  R ≠ A ∧ R ≠ N ∧ R ≠ S ∧ R ≠ P ∧ R ≠ O ∧ R ≠ B ∧ R ≠ K ∧
  A ≠ N ∧ A ≠ S ∧ A ≠ P ∧ A ≠ O ∧ A ≠ B ∧ A ≠ K ∧
  N ≠ S ∧ N ≠ P ∧ N ≠ O ∧ N ≠ B ∧ N ≠ K ∧
  S ≠ P ∧ S ≠ O ∧ S ≠ B ∧ S ≠ K ∧
  P ≠ O ∧ P ≠ B ∧ P ≠ K ∧
  O ≠ B ∧ O ≠ K ∧
  B ≠ K

theorem transportation_puzzle :
  ∃! (T R A N S P O B K : ℕ), is_valid_assignment T R A N S P O B K :=
sorry

end transportation_puzzle_l2087_208716


namespace trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l2087_208792

-- Define the geometric shapes
class ConvexPolygon
class Polygon extends ConvexPolygon
class Trapezoid extends ConvexPolygon
class Rhombus extends ConvexPolygon
class Triangle extends Polygon
class Parallelogram extends Polygon
class Rectangle extends Polygon
class Circle

-- Define properties
def hasExteriorAngleSum360 (shape : Type) : Prop := sorry
def lineIntersectsTwice (shape : Type) : Prop := sorry
def hasCentralSymmetry (shape : Type) : Prop := sorry

-- Theorem statements
theorem trapezoid_rhombus_properties :
  (hasExteriorAngleSum360 Trapezoid ∧ hasExteriorAngleSum360 Rhombus) ∧
  (lineIntersectsTwice Trapezoid ∧ lineIntersectsTwice Rhombus) := by sorry

theorem triangle_parallelogram_properties :
  (hasExteriorAngleSum360 Triangle ∧ hasExteriorAngleSum360 Parallelogram) ∧
  (lineIntersectsTwice Triangle ∧ lineIntersectsTwice Parallelogram) := by sorry

theorem rectangle_circle_symmetry :
  hasCentralSymmetry Rectangle ∧ hasCentralSymmetry Circle := by sorry

end trapezoid_rhombus_properties_triangle_parallelogram_properties_rectangle_circle_symmetry_l2087_208792


namespace max_soap_boxes_in_carton_l2087_208728

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the carton -/
def carton_dimensions : Dimensions := ⟨30, 42, 60⟩

/-- The dimensions of a soap box -/
def soap_box_dimensions : Dimensions := ⟨7, 6, 5⟩

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (volume carton_dimensions) / (volume soap_box_dimensions) = 360 := by
  sorry

end max_soap_boxes_in_carton_l2087_208728


namespace quadratic_two_roots_l2087_208771

theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end quadratic_two_roots_l2087_208771


namespace dutch_americans_window_seats_fraction_l2087_208770

/-- The fraction of Dutch Americans who got window seats on William's bus -/
theorem dutch_americans_window_seats_fraction
  (total_people : ℕ)
  (dutch_fraction : ℚ)
  (dutch_american_fraction : ℚ)
  (dutch_americans_with_window_seats : ℕ)
  (h1 : total_people = 90)
  (h2 : dutch_fraction = 3 / 5)
  (h3 : dutch_american_fraction = 1 / 2)
  (h4 : dutch_americans_with_window_seats = 9) :
  (dutch_americans_with_window_seats : ℚ) / (dutch_fraction * dutch_american_fraction * total_people) = 1 / 3 := by
  sorry

#check dutch_americans_window_seats_fraction

end dutch_americans_window_seats_fraction_l2087_208770


namespace graduating_class_size_l2087_208776

theorem graduating_class_size (boys : ℕ) (girls : ℕ) : 
  boys = 127 → 
  girls = boys + 212 → 
  boys + girls = 466 := by sorry

end graduating_class_size_l2087_208776


namespace point_coordinates_l2087_208783

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := |y|

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem point_coordinates :
  ∀ (x y : ℝ),
    fourth_quadrant x y →
    distance_to_x_axis y = 2 →
    distance_to_y_axis x = 4 →
    x = 4 ∧ y = -2 := by
  sorry

end point_coordinates_l2087_208783


namespace A_union_B_eq_A_l2087_208773

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {x | 0 < x ∧ x < Real.exp 1}

theorem A_union_B_eq_A : A ∪ B = A := by sorry

end A_union_B_eq_A_l2087_208773


namespace function_domain_range_nonempty_function_range_determined_single_element_domain_range_l2087_208799

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1: The domain and range of a function are both non-empty sets
theorem function_domain_range_nonempty {α β : Type} (f : Function α β) :
  Nonempty α ∧ Nonempty β :=
sorry

-- Statement 2: Once the domain and the rule of correspondence are determined,
-- the range of the function is also determined
theorem function_range_determined {α β : Type} (f g : Function α β) :
  (∀ x : α, f x = g x) → Set.range f = Set.range g :=
sorry

-- Statement 3: If there is only one element in the domain of a function,
-- then there is also only one element in its range
theorem single_element_domain_range {α β : Type} (f : Function α β) :
  (∃! x : α, True) → (∃! y : β, ∃ x : α, f x = y) :=
sorry

end function_domain_range_nonempty_function_range_determined_single_element_domain_range_l2087_208799


namespace complex_sum_product_nonzero_l2087_208750

theorem complex_sum_product_nonzero (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) (h₄ : Complex.abs z₄ = 1)
  (n₁ : z₁ ≠ 1) (n₂ : z₂ ≠ 1) (n₃ : z₃ ≠ 1) (n₄ : z₄ ≠ 1) :
  3 - z₁ - z₂ - z₃ - z₄ + z₁ * z₂ * z₃ * z₄ ≠ 0 := by
  sorry

end complex_sum_product_nonzero_l2087_208750


namespace k_condition_necessary_not_sufficient_l2087_208764

/-- Defines the condition for k -/
def k_condition (k : ℝ) : Prop := 7 < k ∧ k < 9

/-- Defines the equation of the conic section -/
def is_conic_equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (9 - k) + y^2 / (k - 7) = 1

/-- Defines the conditions for the equation to represent an ellipse -/
def is_ellipse_equation (k : ℝ) : Prop :=
  9 - k > 0 ∧ k - 7 > 0 ∧ 9 - k ≠ k - 7

/-- Theorem stating that k_condition is necessary but not sufficient for is_ellipse_equation -/
theorem k_condition_necessary_not_sufficient :
  (∀ k, is_ellipse_equation k → k_condition k) ∧
  ¬(∀ k, k_condition k → is_ellipse_equation k) :=
sorry

end k_condition_necessary_not_sufficient_l2087_208764


namespace special_triangle_properties_l2087_208732

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  c_eq : c = 7/2
  area_eq : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2
  tan_eq : Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

/-- Theorem about the properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.C = π/3 ∧ t.a + t.b = 11/2 := by
  sorry

end special_triangle_properties_l2087_208732


namespace circle_tangency_radius_l2087_208701

theorem circle_tangency_radius (r_P r_Q r_R : ℝ) : 
  r_P = 4 ∧ 
  r_Q = 4 * r_R ∧ 
  r_P > r_Q ∧ 
  r_P > r_R ∧
  r_Q > r_R ∧
  r_P = r_Q + r_R →
  r_Q = 16 ∧ 
  r_Q = Real.sqrt 256 - 0 := by
sorry

end circle_tangency_radius_l2087_208701


namespace g_zero_value_l2087_208714

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 6

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -18

-- Theorem to prove
theorem g_zero_value : g.coeff 0 = -3 := by sorry

end g_zero_value_l2087_208714


namespace count_integers_satisfying_equation_l2087_208757

def count_satisfying_integers (lower upper : ℕ) : ℕ :=
  (upper - lower + 1) / 4 + 1

theorem count_integers_satisfying_equation : 
  count_satisfying_integers 1 2002 = 501 := by
  sorry

end count_integers_satisfying_equation_l2087_208757


namespace intersection_M_N_l2087_208789

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x > 2}

theorem intersection_M_N : M ∩ N = {4} := by
  sorry

end intersection_M_N_l2087_208789


namespace digital_earth_capabilities_l2087_208793

structure DigitalEarth where
  simulate_environment : Bool
  monitor_crops : Bool
  predict_submersion : Bool
  simulate_past : Bool
  predict_future : Bool

theorem digital_earth_capabilities (de : DigitalEarth) :
  de.simulate_environment ∧
  de.monitor_crops ∧
  de.predict_submersion ∧
  de.simulate_past →
  ¬ de.predict_future :=
by sorry

end digital_earth_capabilities_l2087_208793


namespace quadrilateral_inequality_quadrilateral_inequality_equality_condition_l2087_208735

/-- Theorem: Quadrilateral Inequality
For any quadrilateral with sides a₁, a₂, a₃, a₄ and semi-perimeter s,
the sum of reciprocals of (aᵢ + s) is less than or equal to 2/9 times
the sum of reciprocals of square roots of (s-aᵢ)(s-aⱼ) for all pairs i,j. -/
theorem quadrilateral_inequality (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s)) ≤ 
  (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₃) * (s - a₄))) :=
by sorry

/-- Corollary: Equality condition for the quadrilateral inequality -/
theorem quadrilateral_inequality_equality_condition (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s) = 
   (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₃) * (s - a₄)))) ↔ 
  (a₁ = a₂ ∧ a₂ = a₃ ∧ a₃ = a₄) :=
by sorry

end quadrilateral_inequality_quadrilateral_inequality_equality_condition_l2087_208735


namespace pirate_coins_l2087_208760

theorem pirate_coins (x : ℚ) : 
  (3/7 * x + 0.51 * (4/7 * x) = (2.04/7) * x) →
  ((2.04/7) * x - (1.96/7) * x = 8) →
  x = 700 :=
by sorry

end pirate_coins_l2087_208760


namespace zhu_shijie_wine_problem_l2087_208766

/-- The amount of wine in the jug after visiting n taverns and meeting n friends -/
def wine_amount (initial : ℝ) (n : ℕ) : ℝ :=
  (2^n) * initial - (2^n - 1)

theorem zhu_shijie_wine_problem :
  ∃ (initial : ℝ), initial > 0 ∧ wine_amount initial 3 = 0 ∧ initial = 0.875 := by
  sorry

end zhu_shijie_wine_problem_l2087_208766


namespace not_perfect_cube_l2087_208713

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℤ, 2^(2^n) + 1 = k^3 := by
  sorry

end not_perfect_cube_l2087_208713


namespace squirrel_travel_time_l2087_208712

/-- Proves that a squirrel traveling 3 miles at 6 miles per hour takes 30 minutes -/
theorem squirrel_travel_time :
  let speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 3 -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by
  sorry

end squirrel_travel_time_l2087_208712


namespace ratio_problem_l2087_208727

theorem ratio_problem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_ratio : ∃ (k : ℝ), a = 6*k ∧ b = 3*k ∧ c = k) :
  3 * b^2 / (2 * a^2 + b * c) = 9 / 25 := by
  sorry

end ratio_problem_l2087_208727


namespace sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2087_208707

/-- The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon : ℝ :=
  360

/-- A polygon is a closed plane figure with straight sides. -/
def Polygon : Type := sorry

/-- A regular polygon is a polygon with all sides and angles equal. -/
def RegularPolygon (p : Polygon) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def Pentagon (p : Polygon) : Prop := sorry

/-- The sum of the exterior angles of any polygon is constant. -/
axiom sum_exterior_angles_constant (p : Polygon) : ℝ

/-- The sum of the exterior angles of any polygon is 360 degrees. -/
axiom sum_exterior_angles_360 (p : Polygon) : sum_exterior_angles_constant p = 360

/-- Theorem: The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon_proof (p : Polygon) 
  (h1 : RegularPolygon p) (h2 : Pentagon p) : 
  sum_exterior_angles_constant p = 360 := by
  sorry

end sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2087_208707


namespace vendor_apples_thrown_away_l2087_208742

/-- Calculates the percentage of apples thrown away given the initial quantity and selling/discarding percentages --/
def apples_thrown_away (initial_quantity : ℕ) (sell_day1 sell_day2 discard_day1 : ℚ) : ℚ :=
  let remaining_after_sell1 := initial_quantity * (1 - sell_day1)
  let discarded_day1 := remaining_after_sell1 * discard_day1
  let remaining_after_discard1 := remaining_after_sell1 - discarded_day1
  let remaining_after_sell2 := remaining_after_discard1 * (1 - sell_day2)
  (discarded_day1 + remaining_after_sell2) / initial_quantity * 100

theorem vendor_apples_thrown_away :
  apples_thrown_away 100 (30/100) (50/100) (20/100) = 42 :=
by sorry

end vendor_apples_thrown_away_l2087_208742


namespace mikes_muffins_l2087_208700

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes needed to pack all muffins -/
def boxes : ℕ := 8

/-- The total number of muffins Mike has -/
def total_muffins : ℕ := boxes * dozen

theorem mikes_muffins : total_muffins = 96 := by
  sorry

end mikes_muffins_l2087_208700


namespace system_solution_arithmetic_progression_l2087_208762

/-- 
Given a system of equations:
  x + y + m*z = a
  x + m*y + z = b
  m*x + y + z = c
This theorem states that for m ≠ 1 and m ≠ -2, the system has a unique solution (x, y, z) 
in arithmetic progression if and only if a, b, c are in arithmetic progression.
-/
theorem system_solution_arithmetic_progression 
  (m a b c : ℝ) (hm1 : m ≠ 1) (hm2 : m ≠ -2) :
  (∃! x y z : ℝ, x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c ∧ 
   2*y = x + z) ↔ 2*b = a + c :=
sorry

end system_solution_arithmetic_progression_l2087_208762


namespace correct_operations_l2087_208780

theorem correct_operations (x y : ℝ) (h : x ≠ y) :
  ((-3 * x * y) ^ 2 = 9 * x^2 * y^2) ∧
  ((x - y) / (2 * x * y - x^2 - y^2) = 1 / (y - x)) := by
  sorry

end correct_operations_l2087_208780


namespace cost_price_calculation_l2087_208768

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 400)
  (h2 : profit_percentage = 25) : 
  ∃ (cost_price : ℝ), 
    cost_price = 320 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end cost_price_calculation_l2087_208768


namespace geometric_series_sum_l2087_208711

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ :=
  15 / (1 - r)

/-- For -1 < b < 1, if T(b)T(-b) = 3240, then T(b) + T(-b) = 432 -/
theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) 
    (h : T b * T (-b) = 3240) : T b + T (-b) = 432 := by
  sorry

end geometric_series_sum_l2087_208711


namespace function_value_at_negative_one_l2087_208723

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_value_at_negative_one
  (h1 : ∀ x : ℝ, f (x + 2009) = -f (x + 2008))
  (h2 : f 2009 = -2009) :
  f (-1) = -2009 := by
  sorry

end function_value_at_negative_one_l2087_208723


namespace complex_square_one_plus_i_l2087_208726

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end complex_square_one_plus_i_l2087_208726


namespace N_is_composite_l2087_208785

/-- N is defined as 7 × 9 × 13 + 2020 × 2018 × 2014 -/
def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

/-- Theorem stating that N is composite -/
theorem N_is_composite : ¬ Nat.Prime N := by sorry

end N_is_composite_l2087_208785


namespace margaret_swimming_time_l2087_208749

/-- Billy's swimming times for different parts of the race in seconds -/
def billy_times : List ℕ := [120, 240, 60, 150]

/-- The time difference between Billy and Margaret in seconds -/
def time_difference : ℕ := 30

/-- Calculate the total time Billy spent swimming -/
def billy_total_time : ℕ := billy_times.sum

/-- Calculate Margaret's total swimming time in seconds -/
def margaret_time_seconds : ℕ := billy_total_time + time_difference

/-- Convert seconds to minutes -/
def seconds_to_minutes (seconds : ℕ) : ℕ := seconds / 60

theorem margaret_swimming_time :
  seconds_to_minutes margaret_time_seconds = 10 := by
  sorry

end margaret_swimming_time_l2087_208749


namespace profit_equation_l2087_208733

/-- Given a profit equation P = (1/m)S - (1/n)C, prove that P = (m-n)/(mn) * S -/
theorem profit_equation (m n : ℝ) (m_ne_zero : m ≠ 0) (n_ne_zero : n ≠ 0) :
  ∀ (S C P : ℝ), P = (1/m) * S - (1/n) * C → P = (m-n)/(m*n) * S :=
by sorry

end profit_equation_l2087_208733


namespace cars_meeting_time_l2087_208763

/-- The time when two cars meet on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) :
  highway_length = 600 →
  speed1 = 65 →
  speed2 = 75 →
  (highway_length / (speed1 + speed2) : ℝ) = 30 / 7 :=
by sorry

end cars_meeting_time_l2087_208763


namespace min_jumps_proof_l2087_208715

/-- The distance of each jump in millimeters -/
def jump_distance : ℝ := 19

/-- The distance between points A and B in centimeters -/
def total_distance : ℝ := 1812

/-- The minimum number of jumps required -/
def min_jumps : ℕ := 954

/-- Theorem stating the minimum number of jumps required -/
theorem min_jumps_proof :
  ∃ (n : ℕ), n = min_jumps ∧ 
  (n : ℝ) * jump_distance ≥ total_distance * 10 ∧
  ∀ (m : ℕ), (m : ℝ) * jump_distance ≥ total_distance * 10 → m ≥ n :=
sorry

end min_jumps_proof_l2087_208715


namespace greatest_common_divisor_420_90_under_50_l2087_208722

theorem greatest_common_divisor_420_90_under_50 : 
  ∀ n : ℕ, n ∣ 420 ∧ n < 50 ∧ n ∣ 90 → n ≤ 30 :=
by
  sorry

end greatest_common_divisor_420_90_under_50_l2087_208722


namespace completing_square_equiv_l2087_208758

theorem completing_square_equiv (x : ℝ) : 
  x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by sorry

end completing_square_equiv_l2087_208758


namespace point_movement_l2087_208738

theorem point_movement (A B : ℝ × ℝ) : 
  A = (-3, 2) → 
  B.1 = A.1 + 1 → 
  B.2 = A.2 - 2 → 
  B = (-2, 0) := by
sorry

end point_movement_l2087_208738


namespace target_probabilities_l2087_208755

/-- Probability of hitting a target -/
structure TargetProbability where
  prob : ℚ
  prob_nonneg : 0 ≤ prob
  prob_le_one : prob ≤ 1

/-- Model for the target shooting scenario -/
structure TargetScenario where
  A : TargetProbability
  B : TargetProbability

/-- Given scenario with person A and B's probabilities -/
def given_scenario : TargetScenario :=
  { A := { prob := 3/4, prob_nonneg := by norm_num, prob_le_one := by norm_num },
    B := { prob := 4/5, prob_nonneg := by norm_num, prob_le_one := by norm_num } }

/-- Probability that A hits and B misses after one shot each -/
def prob_A_hits_B_misses (s : TargetScenario) : ℚ :=
  s.A.prob * (1 - s.B.prob)

/-- Probability of k successes in n independent trials -/
def binomial_prob (p : ℚ) (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1-p)^(n-k)

/-- Probability that A and B have equal hits after two shots each -/
def prob_equal_hits (s : TargetScenario) : ℚ :=
  (binomial_prob s.A.prob 2 0) * (binomial_prob s.B.prob 2 0) +
  (binomial_prob s.A.prob 2 1) * (binomial_prob s.B.prob 2 1) +
  (binomial_prob s.A.prob 2 2) * (binomial_prob s.B.prob 2 2)

theorem target_probabilities (s : TargetScenario := given_scenario) :
  (prob_A_hits_B_misses s = 3/20) ∧
  (prob_equal_hits s = 193/400) := by
  sorry

end target_probabilities_l2087_208755


namespace total_sum_is_71_rupees_l2087_208724

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let sum_20_paise := (coins_20_paise : ℚ) * (20 : ℚ) / 100
  let sum_25_paise := (coins_25_paise : ℚ) * (25 : ℚ) / 100
  sum_20_paise + sum_25_paise

/-- Theorem stating that given 324 total coins with 200 coins of 20 paise, the total sum is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_rupees 324 200 = 71 := by
  sorry

end total_sum_is_71_rupees_l2087_208724


namespace largest_integer_less_than_100_remainder_5_when_divided_by_8_l2087_208774

theorem largest_integer_less_than_100_remainder_5_when_divided_by_8 : 
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end largest_integer_less_than_100_remainder_5_when_divided_by_8_l2087_208774


namespace matt_bike_ride_distance_l2087_208784

theorem matt_bike_ride_distance 
  (distance_to_first_sign : ℕ)
  (distance_between_signs : ℕ)
  (distance_after_second_sign : ℕ)
  (h1 : distance_to_first_sign = 350)
  (h2 : distance_between_signs = 375)
  (h3 : distance_after_second_sign = 275) :
  distance_to_first_sign + distance_between_signs + distance_after_second_sign = 1000 :=
by sorry

end matt_bike_ride_distance_l2087_208784


namespace z_in_first_quadrant_l2087_208746

def complex_i : ℂ := Complex.I

theorem z_in_first_quadrant (z : ℂ) (h : (1 + complex_i) * z = 1 - 2 * complex_i^3) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end z_in_first_quadrant_l2087_208746


namespace fence_overlap_calculation_l2087_208736

theorem fence_overlap_calculation (num_planks : ℕ) (plank_length : ℝ) (total_length : ℝ) 
  (h1 : num_planks = 25)
  (h2 : plank_length = 30)
  (h3 : total_length = 690) :
  ∃ overlap : ℝ, 
    overlap = 2.5 ∧ 
    total_length = (13 * plank_length) + (12 * (plank_length - 2 * overlap)) :=
by sorry

end fence_overlap_calculation_l2087_208736


namespace bridge_anchor_ratio_l2087_208705

/-- Proves that the ratio of concrete needed for each bridge anchor is 1:1 --/
theorem bridge_anchor_ratio
  (total_concrete : ℕ)
  (roadway_concrete : ℕ)
  (one_anchor_concrete : ℕ)
  (pillar_concrete : ℕ)
  (h1 : total_concrete = 4800)
  (h2 : roadway_concrete = 1600)
  (h3 : one_anchor_concrete = 700)
  (h4 : pillar_concrete = 1800)
  : (one_anchor_concrete : ℚ) / one_anchor_concrete = 1 := by
  sorry

#check bridge_anchor_ratio

end bridge_anchor_ratio_l2087_208705


namespace negation_of_universal_statement_l2087_208740

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end negation_of_universal_statement_l2087_208740


namespace ellipse_eccentricity_l2087_208741

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if the length of the minor axis is equal to the focal length,
    then the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let c := Real.sqrt (a^2 - b^2)
  2 * b = 2 * c → Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 := by
  sorry

end ellipse_eccentricity_l2087_208741


namespace log_inequality_equiv_range_l2087_208752

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_equiv_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end log_inequality_equiv_range_l2087_208752


namespace range_of_a_l2087_208759

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 1 → x^2 + 2*x - a > 0) → a < 3 := by
  sorry

end range_of_a_l2087_208759


namespace polynomial_equality_l2087_208717

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end polynomial_equality_l2087_208717


namespace intersection_condition_l2087_208730

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = -p.1^2 + m*p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3 ∧ 0 ≤ p.1 ∧ p.1 ≤ 3}

-- Define the condition for exactly one intersection
def exactly_one_intersection (m : ℝ) : Prop :=
  ∃! p, p ∈ A m ∩ B

-- State the theorem
theorem intersection_condition (m : ℝ) :
  exactly_one_intersection m ↔ (m = 3 ∨ m > 10/3) := by
  sorry

end intersection_condition_l2087_208730


namespace consecutive_negative_integers_product_2850_l2087_208796

theorem consecutive_negative_integers_product_2850 :
  ∃ (n : ℤ), n < 0 ∧ n * (n + 1) = 2850 → (n + (n + 1)) = -107 := by
  sorry

end consecutive_negative_integers_product_2850_l2087_208796


namespace equal_money_after_transfer_l2087_208791

/-- Given that Lucy originally has $20 and Linda has $10, prove that if Lucy gives $5 to Linda,
    they will have the same amount of money. -/
theorem equal_money_after_transfer (lucy_initial : ℕ) (linda_initial : ℕ) (transfer_amount : ℕ) : 
  lucy_initial = 20 →
  linda_initial = 10 →
  transfer_amount = 5 →
  lucy_initial - transfer_amount = linda_initial + transfer_amount :=
by sorry

end equal_money_after_transfer_l2087_208791
