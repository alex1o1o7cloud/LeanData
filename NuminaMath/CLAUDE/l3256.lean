import Mathlib

namespace expression_simplification_l3256_325631

theorem expression_simplification :
  0.7264 * 0.4329 + 0.1235 * 0.3412 + 0.1289 * 0.5634 - 0.3785 * 0.4979 = 0.2407 := by
  sorry

end expression_simplification_l3256_325631


namespace average_of_middle_two_l3256_325683

theorem average_of_middle_two (numbers : Fin 6 → ℝ) 
  (h_total_avg : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 6.40)
  (h_first_two_avg : (numbers 0 + numbers 1) / 2 = 6.2)
  (h_last_two_avg : (numbers 4 + numbers 5) / 2 = 6.9) :
  (numbers 2 + numbers 3) / 2 = 6.1 := by
  sorry

end average_of_middle_two_l3256_325683


namespace imaginary_power_sum_l3256_325603

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end imaginary_power_sum_l3256_325603


namespace three_fifths_of_five_times_nine_l3256_325650

theorem three_fifths_of_five_times_nine : (3 : ℚ) / 5 * (5 * 9) = 27 := by sorry

end three_fifths_of_five_times_nine_l3256_325650


namespace arithmetic_seq_2016_l3256_325621

/-- An arithmetic sequence with common difference 2 and a_2007 = 2007 -/
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = 2) ∧ 
  (a 2007 = 2007)

theorem arithmetic_seq_2016 (a : ℕ → ℕ) (h : arithmetic_seq a) : 
  a 2016 = 2025 := by
  sorry

end arithmetic_seq_2016_l3256_325621


namespace max_value_theorem_l3256_325637

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 3) :
  (1/2 : ℝ)*x + y ≤ Real.sqrt 6 / 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*y₀^2 = 3 ∧ (1/2 : ℝ)*x₀ + y₀ = Real.sqrt 6 / 2 :=
by sorry

end max_value_theorem_l3256_325637


namespace sin_alpha_plus_pi_fourth_l3256_325653

theorem sin_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.sin (2 * α) = 1 / 2) : 
  Real.sin (α + Real.pi / 4) = Real.sqrt 3 / 2 := by
  sorry

end sin_alpha_plus_pi_fourth_l3256_325653


namespace notebook_distribution_ratio_l3256_325694

/-- Given a class where notebooks are distributed equally among children,
    prove that the ratio of notebooks per child to the number of children is 1/8 -/
theorem notebook_distribution_ratio 
  (C : ℕ) -- number of children
  (N : ℕ) -- number of notebooks per child
  (h1 : C * N = 512) -- total notebooks distributed
  (h2 : (C / 2) * 16 = 512) -- if children halved, each gets 16
  : N / C = 1 / 8 := by
  sorry

end notebook_distribution_ratio_l3256_325694


namespace midpoint_trajectory_l3256_325685

/-- The trajectory of the midpoint Q between a point P on the unit circle and a fixed point M -/
theorem midpoint_trajectory (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1 = (P.1 + 2) / 2 ∧ Q.2 = P.2 / 2) →  -- Q is the midpoint of PM where M is (2, 0)
  (Q.1 - 1)^2 + Q.2^2 = 1/4 := by
  sorry

end midpoint_trajectory_l3256_325685


namespace intersection_M_N_l3256_325605

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_M_N_l3256_325605


namespace line_passes_through_fixed_point_l3256_325672

/-- A line passing through a fixed point for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by sorry

end line_passes_through_fixed_point_l3256_325672


namespace min_values_xy_and_x_plus_y_l3256_325644

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + 8 * y₀ - x₀ * y₀ = 0 ∧ x₀ * y₀ = 64) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' * y' ≥ 64) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 2 * x₁ + 8 * y₁ - x₁ * y₁ = 0 ∧ x₁ + y₁ = 18) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' + y' ≥ 18) :=
by sorry

end min_values_xy_and_x_plus_y_l3256_325644


namespace point_symmetric_range_l3256_325673

/-- 
Given a point P(a+1, 2a-3) that is symmetric about the x-axis and lies in the first quadrant,
prove that the range of a is (-1, 3/2).
-/
theorem point_symmetric_range (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 1 ∧ P.2 = 2*a - 3 ∧ P.1 > 0 ∧ P.2 > 0) ↔ 
  -1 < a ∧ a < 3/2 := by sorry

end point_symmetric_range_l3256_325673


namespace mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l3256_325632

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (((6 : ℝ) + 10 + 14 + 22) / 4 = (15 + y) / 2) → y = 11

-- The proof is omitted
theorem mean_equality_implies_y_value_proof : mean_equality_implies_y_value 11 := by
  sorry

end mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l3256_325632


namespace first_half_chop_count_l3256_325642

/-- The number of trees that need to be planted for each tree chopped down -/
def replantRatio : ℕ := 3

/-- The number of trees chopped down in the second half of the year -/
def secondHalfChop : ℕ := 300

/-- The total number of trees that need to be planted -/
def totalPlant : ℕ := 1500

/-- The number of trees chopped down in the first half of the year -/
def firstHalfChop : ℕ := (totalPlant - replantRatio * secondHalfChop) / replantRatio

theorem first_half_chop_count : firstHalfChop = 200 := by
  sorry

end first_half_chop_count_l3256_325642


namespace giraffe_ratio_l3256_325626

theorem giraffe_ratio (total_giraffes : ℕ) (difference : ℕ) : 
  total_giraffes = 300 →
  difference = 290 →
  total_giraffes = (total_giraffes - difference) + difference →
  (total_giraffes : ℚ) / (total_giraffes - difference) = 30 := by
  sorry

end giraffe_ratio_l3256_325626


namespace cube_pyramid_sum_l3256_325657

/-- Represents a three-dimensional shape --/
structure Shape3D where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- A cube --/
def cube : Shape3D :=
  { edges := 12, corners := 8, faces := 6 }

/-- A square pyramid --/
def square_pyramid : Shape3D :=
  { edges := 8, corners := 5, faces := 5 }

/-- The shape formed by placing a square pyramid on one face of a cube --/
def cube_with_pyramid : Shape3D :=
  { edges := cube.edges + 4, -- 4 new edges from pyramid apex
    corners := cube.corners + 1, -- 1 new corner (pyramid apex)
    faces := cube.faces + square_pyramid.faces - 1 } -- -1 for shared base

/-- The sum of edges, corners, and faces of the combined shape --/
def combined_sum (s : Shape3D) : ℕ :=
  s.edges + s.corners + s.faces

/-- Theorem stating that the sum of edges, corners, and faces of the combined shape is 34 --/
theorem cube_pyramid_sum :
  combined_sum cube_with_pyramid = 34 := by
  sorry


end cube_pyramid_sum_l3256_325657


namespace corner_difference_divisible_by_six_l3256_325682

/-- A 9x9 table filled with numbers from 1 to 81 -/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Check if two cells are adjacent -/
def adjacent (i j k l : Fin 9) : Prop :=
  (i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ (j = l ∧ (i = k + 1 ∨ i + 1 = k))

/-- Check if a number is in a corner cell -/
def isCorner (i j : Fin 9) : Prop :=
  (i = 0 ∨ i = 8) ∧ (j = 0 ∨ j = 8)

/-- The main theorem -/
theorem corner_difference_divisible_by_six (t : Table) 
  (h1 : ∀ i j k l, adjacent i j k l → (t i j : ℕ) + 3 = t k l ∨ (t i j : ℕ) = (t k l : ℕ) + 3)
  (h2 : ∀ i j k l, i ≠ k ∨ j ≠ l → t i j ≠ t k l) :
  ∃ i j k l, isCorner i j ∧ isCorner k l ∧ 
    (∃ m : ℕ, (t i j : ℕ) - (t k l : ℕ) = 6 * m ∨ (t k l : ℕ) - (t i j : ℕ) = 6 * m) :=
sorry

end corner_difference_divisible_by_six_l3256_325682


namespace clothing_tax_rate_l3256_325656

theorem clothing_tax_rate (total_spent : ℝ) (clothing_spent : ℝ) (food_spent : ℝ) (other_spent : ℝ)
  (clothing_tax : ℝ) (other_tax : ℝ) (total_tax : ℝ) :
  clothing_spent = 0.4 * total_spent →
  food_spent = 0.3 * total_spent →
  other_spent = 0.3 * total_spent →
  other_tax = 0.08 * other_spent →
  total_tax = 0.04 * total_spent →
  total_tax = clothing_tax + other_tax →
  clothing_tax / clothing_spent = 0.04 :=
by sorry

end clothing_tax_rate_l3256_325656


namespace original_price_of_meat_pack_original_price_is_40_l3256_325610

/-- The original price of a 4 pack of fancy, sliced meat, given rush delivery conditions -/
theorem original_price_of_meat_pack : ℝ :=
  let rush_delivery_factor : ℝ := 1.3
  let price_with_rush : ℝ := 13
  let pack_size : ℕ := 4
  let single_meat_price : ℝ := price_with_rush / rush_delivery_factor
  pack_size * single_meat_price

/-- Proof that the original price of the 4 pack is $40 -/
theorem original_price_is_40 : original_price_of_meat_pack = 40 := by
  sorry

end original_price_of_meat_pack_original_price_is_40_l3256_325610


namespace no_cube_sum_three_consecutive_squares_l3256_325645

theorem no_cube_sum_three_consecutive_squares :
  ¬ ∃ (x y : ℤ), x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by sorry

end no_cube_sum_three_consecutive_squares_l3256_325645


namespace circumscribed_sphere_surface_area_main_theorem_l3256_325624

/-- Represents a tetrahedron A-BCD with specific properties -/
structure Tetrahedron where
  /-- Base BCD is an equilateral triangle with side length 2 -/
  base_side_length : ℝ
  base_is_equilateral : base_side_length = 2
  /-- Projection of vertex A onto base BCD is the center of triangle BCD -/
  vertex_projection_is_center : Bool
  /-- E is the midpoint of side BC -/
  e_is_midpoint : Bool
  /-- Sine of angle formed by line AE with base BCD is 2√2 -/
  sine_angle_ae_base : ℝ
  sine_angle_ae_base_value : sine_angle_ae_base = 2 * Real.sqrt 2

/-- The surface area of the circumscribed sphere of the tetrahedron is 6π -/
theorem circumscribed_sphere_surface_area (t : Tetrahedron) : ℝ := by
  sorry

/-- Main theorem: The surface area of the circumscribed sphere is 6π -/
theorem main_theorem (t : Tetrahedron) : circumscribed_sphere_surface_area t = 6 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_main_theorem_l3256_325624


namespace exists_fixed_point_l3256_325698

variable {α : Type*} [Finite α]

def IsIncreasing (f : Set α → Set α) : Prop :=
  ∀ X Y : Set α, X ⊆ Y → f X ⊆ f Y

theorem exists_fixed_point (f : Set α → Set α) (hf : IsIncreasing f) :
    ∃ H₀ : Set α, f H₀ = H₀ := by
  sorry

end exists_fixed_point_l3256_325698


namespace pencil_distribution_l3256_325686

theorem pencil_distribution (n : Nat) (k : Nat) : 
  n = 6 → k = 3 → (Nat.choose (n - k + k - 1) (k - 1)) = 10 := by
  sorry

end pencil_distribution_l3256_325686


namespace percent_equality_l3256_325688

theorem percent_equality (x : ℝ) : (70 / 100 * 600 = 40 / 100 * x) → x = 1050 := by
  sorry

end percent_equality_l3256_325688


namespace system_solution_l3256_325677

theorem system_solution (a : ℚ) :
  (∃! x y : ℚ, 2*x + 3*y = 5 ∧ x - y = 2 ∧ x + 4*y = a) ↔ a = 3 :=
by sorry

end system_solution_l3256_325677


namespace sum_of_products_l3256_325654

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 250)
  (h2 : a + b + c = 16) :
  a*b + b*c + c*a = 3 := by sorry

end sum_of_products_l3256_325654


namespace factorization_valid_l3256_325693

theorem factorization_valid (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end factorization_valid_l3256_325693


namespace smallest_equal_probability_sum_l3256_325660

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The sum we want to compare with -/
def target_sum : ℕ := 1504

/-- The function to calculate the transformed sum -/
def transformed_sum (n : ℕ) : ℕ := 9 * n - target_sum

/-- The proposition that S is the smallest possible value satisfying the conditions -/
theorem smallest_equal_probability_sum : 
  ∃ (n : ℕ), n * sides ≥ target_sum ∧ 
  ∀ (m : ℕ), m < transformed_sum n → 
  ¬(∃ (k : ℕ), k * sides ≥ target_sum ∧ 
    transformed_sum k = m) :=
sorry

end smallest_equal_probability_sum_l3256_325660


namespace cold_production_time_proof_l3256_325697

/-- The time (in minutes) it takes to produce each pot when the machine is cold. -/
def cold_production_time : ℝ := 6

/-- The time (in minutes) it takes to produce each pot when the machine is warm. -/
def warm_production_time : ℝ := 5

/-- The number of additional pots produced in the last hour compared to the first. -/
def additional_pots : ℕ := 2

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

theorem cold_production_time_proof :
  cold_production_time = 6 ∧
  warm_production_time = 5 ∧
  additional_pots = 2 ∧
  minutes_per_hour / cold_production_time + additional_pots = minutes_per_hour / warm_production_time :=
by sorry

end cold_production_time_proof_l3256_325697


namespace married_men_fraction_l3256_325699

-- Define the structure of the gathering
structure Gathering where
  single_women : ℕ
  married_couples : ℕ

-- Define the probability of a woman being single
def prob_single_woman (g : Gathering) : ℚ :=
  g.single_women / (g.single_women + g.married_couples)

-- Define the fraction of married men in the gathering
def fraction_married_men (g : Gathering) : ℚ :=
  g.married_couples / (g.single_women + 2 * g.married_couples)

-- Theorem statement
theorem married_men_fraction (g : Gathering) :
  prob_single_woman g = 1/3 → fraction_married_men g = 2/5 := by
  sorry

end married_men_fraction_l3256_325699


namespace complex_number_solution_l3256_325663

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_solution (z : ℂ) 
  (h1 : is_purely_imaginary (z - 1))
  (h2 : is_purely_imaginary ((z + 1)^2 - 8*I)) :
  z = 1 - 2*I :=
by sorry

end complex_number_solution_l3256_325663


namespace total_books_on_shelves_l3256_325691

theorem total_books_on_shelves (x : ℕ) : 
  (x / 2 + 5 = 2 * (x / 2 - 5)) → x = 30 := by
  sorry

end total_books_on_shelves_l3256_325691


namespace mars_mission_cost_share_l3256_325651

/-- The total cost in billions of dollars to send a person to Mars -/
def total_cost : ℝ := 30

/-- The number of people in millions sharing the cost -/
def number_of_people : ℝ := 300

/-- Each person's share of the cost in dollars -/
def cost_per_person : ℝ := 100

/-- Theorem stating that if the total cost in billions of dollars is shared equally among the given number of people in millions, each person's share is the specified amount in dollars -/
theorem mars_mission_cost_share : 
  (total_cost * 1000) / number_of_people = cost_per_person := by
  sorry

end mars_mission_cost_share_l3256_325651


namespace integer_solutions_l3256_325623

def satisfies_inequalities (x : ℤ) : Prop :=
  (x + 8 : ℚ) / (x + 2 : ℚ) > 2 ∧ Real.log (x - 1 : ℝ) < 1

theorem integer_solutions :
  {x : ℤ | satisfies_inequalities x} = {2, 3} := by sorry

end integer_solutions_l3256_325623


namespace vector_sum_scalar_multiple_l3256_325649

/-- Given two planar vectors a and b, prove that 3a + b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 0) → (3 • a) + b = (-2, 6) := by sorry

end vector_sum_scalar_multiple_l3256_325649


namespace article_cost_price_l3256_325692

theorem article_cost_price (C : ℝ) : C = 400 :=
  by
  have h1 : 1.05 * C - 2 = 0.95 * C * 1.10 := by sorry
  sorry

end article_cost_price_l3256_325692


namespace magnet_to_stuffed_animals_ratio_l3256_325614

-- Define the cost of the magnet
def magnet_cost : ℚ := 3

-- Define the cost of a single stuffed animal
def stuffed_animal_cost : ℚ := 6

-- Define the combined cost of two stuffed animals
def two_stuffed_animals_cost : ℚ := 2 * stuffed_animal_cost

-- Theorem stating the ratio of magnet cost to combined stuffed animals cost
theorem magnet_to_stuffed_animals_ratio :
  magnet_cost / two_stuffed_animals_cost = 1 / 4 := by sorry

end magnet_to_stuffed_animals_ratio_l3256_325614


namespace inequality_proof_l3256_325620

theorem inequality_proof (a b c d k : ℝ) 
  (h1 : |k| < 2) 
  (h2 : a^2 + b^2 - k*a*b = 1) 
  (h3 : c^2 + d^2 - k*c*d = 1) : 
  |a*c - b*d| ≤ 2 / Real.sqrt (4 - k^2) := by
  sorry

end inequality_proof_l3256_325620


namespace even_rows_in_pascal_triangle_l3256_325635

/-- Pascal's triangle row -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Check if a row (excluding endpoints) consists of only even numbers -/
def is_even_row (row : List ℕ) : Bool := sorry

/-- Count of even rows in first n rows of Pascal's triangle (excluding row 0 and 1) -/
def count_even_rows (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 4 even rows in the first 30 rows of Pascal's triangle (excluding row 0 and 1) -/
theorem even_rows_in_pascal_triangle : count_even_rows 30 = 4 := by sorry

end even_rows_in_pascal_triangle_l3256_325635


namespace fourth_power_nested_roots_l3256_325619

theorem fourth_power_nested_roots : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 2)))^4 = 4 + 2 * Real.sqrt 3 := by
  sorry

end fourth_power_nested_roots_l3256_325619


namespace bushes_for_zucchinis_l3256_325609

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 18

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush = target_zucchinis * containers_per_zucchini :=
by sorry

end bushes_for_zucchinis_l3256_325609


namespace quadratic_equation_roots_l3256_325638

theorem quadratic_equation_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*m*x₁ - m - 1 = 0) ∧ 
  (x₂^2 - 2*m*x₂ - m - 1 = 0) :=
sorry

end quadratic_equation_roots_l3256_325638


namespace point_not_in_third_quadrant_l3256_325627

/-- A point P(x, y) on the line y = -x + 1 cannot be in the third quadrant -/
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) := by
  sorry

end point_not_in_third_quadrant_l3256_325627


namespace max_value_f_l3256_325661

def f (x : ℝ) := x * (1 - x)

theorem max_value_f :
  ∃ (m : ℝ), ∀ (x : ℝ), 0 < x ∧ x < 1 → f x ≤ m ∧ (∃ (y : ℝ), 0 < y ∧ y < 1 ∧ f y = m) ∧ m = 1/4 := by
  sorry

end max_value_f_l3256_325661


namespace number_categorization_l3256_325607

def given_numbers : List ℚ := [-10, 2/3, 0, -0.6, 4, -4 - 2/7]

def positive_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x > 0}

def negative_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0}

def integer_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ ∃ n : ℤ, x = n}

def negative_fractions (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0 ∧ ¬∃ n : ℤ, x = n}

theorem number_categorization :
  positive_numbers given_numbers = {2/3, 4} ∧
  negative_numbers given_numbers = {-10, -0.6, -4 - 2/7} ∧
  integer_numbers given_numbers = {-10, 0, 4} ∧
  negative_fractions given_numbers = {-0.6, -4 - 2/7} := by
  sorry

end number_categorization_l3256_325607


namespace distance_from_apex_l3256_325664

/-- A right octagonal pyramid with two parallel cross sections -/
structure OctagonalPyramid where
  /-- Area of the smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of the larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem about the distance of the larger cross section from the apex -/
theorem distance_from_apex (pyramid : OctagonalPyramid)
  (h_area_small : pyramid.area_small = 256 * Real.sqrt 2)
  (h_area_large : pyramid.area_large = 576 * Real.sqrt 2)
  (h_distance : pyramid.distance_between = 10) :
  ∃ (d : ℝ), d = 30 ∧ d > 0 ∧ 
  d * d * pyramid.area_small = (d - pyramid.distance_between) * (d - pyramid.distance_between) * pyramid.area_large :=
sorry

end distance_from_apex_l3256_325664


namespace geometric_sequence_property_l3256_325630

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : a 1 > 0) 
  (h2 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end geometric_sequence_property_l3256_325630


namespace ivanov_family_net_worth_l3256_325662

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The remaining mortgage balance of the Ivanov family in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The remaining car loan balance of the Ivanov family in rubles -/
def car_loan_balance : ℕ := 500000

/-- The debt the Ivanov family owes to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- The total assets of the Ivanov family -/
def total_assets : ℕ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash

/-- The total liabilities of the Ivanov family -/
def total_liabilities : ℕ := mortgage_balance + car_loan_balance + debt_to_relatives

theorem ivanov_family_net_worth :
  ivanov_net_worth = total_assets - total_liabilities := by sorry

end ivanov_family_net_worth_l3256_325662


namespace geometric_sequence_and_sum_l3256_325618

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the arithmetic sequence c_n
def c (n : ℕ) : ℝ := 2 * n + 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := c n - a n

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := n^2 + 3*n - 3^n + 1

theorem geometric_sequence_and_sum :
  (∀ n, a (n + 1) / a n > 1) ∧  -- Common ratio > 1
  a 2 = 6 ∧
  a 1 + a 2 + a 3 = 26 ∧
  (∀ n, c n = a n + b n) ∧
  (∀ n, c (n + 1) - c n = c 2 - c 1) ∧  -- c_n is arithmetic
  b 1 = a 1 ∧
  b 3 = -10 →
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  (∀ n, S n = n^2 + 3*n - 3^n + 1) := by sorry

end geometric_sequence_and_sum_l3256_325618


namespace total_books_l3256_325628

theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) : 
  stu_books = 9 → 
  albert_ratio = 4.5 → 
  stu_books + albert_ratio * stu_books = 49.5 := by
sorry

end total_books_l3256_325628


namespace square_root_of_25_l3256_325678

theorem square_root_of_25 : (Real.sqrt 25) ^ 2 = 25 := by
  sorry

end square_root_of_25_l3256_325678


namespace exactly_one_prob_l3256_325646

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- The events A and B are independent -/
axiom independent : True

/-- The probability that exactly one of A or B occurs -/
def prob_exactly_one : ℝ := (1 - prob_A) * prob_B + prob_A * (1 - prob_B)

/-- Theorem: The probability that exactly one of A or B occurs is 0.5 -/
theorem exactly_one_prob : prob_exactly_one = 0.5 := by
  sorry

end exactly_one_prob_l3256_325646


namespace gcd_from_lcm_and_ratio_l3256_325634

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 200 → A * 5 = B * 2 → Nat.gcd A B = 20 := by
  sorry

end gcd_from_lcm_and_ratio_l3256_325634


namespace smallest_result_is_24_l3256_325680

def S : Finset ℕ := {2, 3, 5, 7, 11, 13}

def isConsecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

def validTriple (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ¬isConsecutive a b ∧ ¬isConsecutive b c ∧ ¬isConsecutive a c

def process (a b c : ℕ) : Finset ℕ :=
  {a * (b + c), b * (a + c), c * (a + b)}

theorem smallest_result_is_24 :
  ∀ a b c : ℕ, validTriple a b c →
    ∃ x ∈ process a b c, x ≥ 24 ∧ ∀ y ∈ process a b c, y ≥ x :=
by sorry

end smallest_result_is_24_l3256_325680


namespace hyperbola_asymptote_l3256_325658

/-- Given a hyperbola with equation x²/m - y²/3 = 1 where m > 0,
    if one of its asymptotic lines is y = (1/2)x, then m = 12 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ (x y : ℝ), x^2 / m - y^2 / 3 = 1 ∧ y = (1/2) * x) → m = 12 := by
  sorry

end hyperbola_asymptote_l3256_325658


namespace adult_tickets_sold_l3256_325696

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧ 
    adult_tickets = 40 := by
  sorry

end adult_tickets_sold_l3256_325696


namespace man_son_age_ratio_l3256_325669

/-- 
Given a man and his son, where:
- The man is 28 years older than his son
- The son's present age is 26 years
Prove that the ratio of their ages in two years will be 2:1
-/
theorem man_son_age_ratio : 
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end man_son_age_ratio_l3256_325669


namespace sine_shift_right_l3256_325647

/-- Shifting a sine function to the right by π/6 units -/
theorem sine_shift_right (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t + π / 6)
  let g (t : ℝ) := f (t - π / 6)
  g x = Real.sin (2 * x - π / 6) := by
  sorry

end sine_shift_right_l3256_325647


namespace divisibility_of_all_ones_number_l3256_325695

/-- A positive integer whose decimal representation contains only ones -/
def AllOnesNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n.val = (10^k.val - 1) / 9

theorem divisibility_of_all_ones_number (N : ℕ+) 
  (h_all_ones : AllOnesNumber N) 
  (h_div_7 : 7 ∣ N.val) : 
  13 ∣ N.val := by
  sorry

end divisibility_of_all_ones_number_l3256_325695


namespace probability_of_stopping_is_43_103_l3256_325616

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightCycle where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the probability of stopping at a traffic light -/
def probabilityOfStopping (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleTime := cycle.red + cycle.green + cycle.yellow
  let stoppingTime := cycle.red + cycle.yellow
  stoppingTime / totalCycleTime

/-- The specific traffic light cycle in the problem -/
def problemCycle : TrafficLightCycle :=
  { red := 40, green := 60, yellow := 3 }

theorem probability_of_stopping_is_43_103 :
  probabilityOfStopping problemCycle = 43 / 103 := by
  sorry

end probability_of_stopping_is_43_103_l3256_325616


namespace some_number_equation_l3256_325617

theorem some_number_equation (x : ℤ) : |x - 8*(3 - 12)| - |5 - 11| = 70 ↔ x = 4 := by
  sorry

end some_number_equation_l3256_325617


namespace pencil_box_sequence_l3256_325613

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 0 = 78) (h2 : a 1 = 87) (h3 : a 2 = 96) (h4 : a 3 = 105)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : a 4 = 114 := by
  sorry

end pencil_box_sequence_l3256_325613


namespace least_addition_for_divisibility_by_five_l3256_325681

theorem least_addition_for_divisibility_by_five (n : ℕ) (h : n = 821562) :
  ∃ k : ℕ, k = 3 ∧ (n + k) % 5 = 0 ∧ ∀ m : ℕ, m < k → (n + m) % 5 ≠ 0 :=
sorry

end least_addition_for_divisibility_by_five_l3256_325681


namespace expression_equals_four_l3256_325641

theorem expression_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = y^2) :
  (x + 1/x) * (y - 1/y) = 4 := by
  sorry

end expression_equals_four_l3256_325641


namespace power_of_ten_multiplication_l3256_325608

theorem power_of_ten_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end power_of_ten_multiplication_l3256_325608


namespace polynomial_root_not_all_real_l3256_325659

theorem polynomial_root_not_all_real (a b c d e : ℝ) :
  2 * a^2 < 5 * b →
  ∃ z : ℂ, z^5 + a*z^4 + b*z^3 + c*z^2 + d*z + e = 0 ∧ z.im ≠ 0 :=
by sorry

end polynomial_root_not_all_real_l3256_325659


namespace sum_of_digits_7ab_l3256_325622

/-- Integer consisting of 1234 sevens in base 10 -/
def a : ℕ := 7 * (10^1234 - 1) / 9

/-- Integer consisting of 1234 twos in base 10 -/
def b : ℕ := 2 * (10^1234 - 1) / 9

/-- Sum of digits in the base 10 representation of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_7ab : sum_of_digits (7 * a * b) = 11100 := by sorry

end sum_of_digits_7ab_l3256_325622


namespace complex_inequality_l3256_325648

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end complex_inequality_l3256_325648


namespace prob_four_green_marbles_l3256_325611

def total_marbles : ℕ := 15
def green_marbles : ℕ := 10
def purple_marbles : ℕ := 5
def total_draws : ℕ := 8
def green_draws : ℕ := 4

theorem prob_four_green_marbles :
  (Nat.choose total_draws green_draws : ℚ) *
  (green_marbles / total_marbles : ℚ) ^ green_draws *
  (purple_marbles / total_marbles : ℚ) ^ (total_draws - green_draws) =
  1120 / 6561 := by sorry

end prob_four_green_marbles_l3256_325611


namespace square_sum_from_difference_and_product_l3256_325643

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by sorry

end square_sum_from_difference_and_product_l3256_325643


namespace angle_measure_in_pentagon_l3256_325679

structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

def is_convex_pentagon (p : Pentagon) : Prop :=
  p.F + p.G + p.H + p.I + p.J = 540

theorem angle_measure_in_pentagon (p : Pentagon) 
  (convex : is_convex_pentagon p)
  (fgh_congruent : p.F = p.G ∧ p.G = p.H)
  (ij_congruent : p.I = p.J)
  (f_less_than_i : p.F + 80 = p.I) :
  p.I = 156 := by
  sorry

end angle_measure_in_pentagon_l3256_325679


namespace monotone_increasing_condition_l3256_325655

/-- The function f(x) = kx - 2ln(x) is monotonically increasing on [1, +∞) iff k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x ≥ 1, Monotone (λ x => k * x - 2 * Real.log x)) ↔ k ≥ 2 := by
sorry

end monotone_increasing_condition_l3256_325655


namespace magnified_tissue_diameter_l3256_325666

/-- Given a circular piece of tissue and an electron microscope, 
    calculates the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem stating that for the given conditions, 
    the magnified diameter is 2 centimeters. -/
theorem magnified_tissue_diameter :
  let actual_diameter : ℝ := 0.002
  let magnification_factor : ℝ := 1000
  magnified_diameter actual_diameter magnification_factor = 2 := by
  sorry

end magnified_tissue_diameter_l3256_325666


namespace eleventhNumberWithSumOfDigits12Is156_l3256_325671

-- Define a function to check if the sum of digits of a number is 12
def sumOfDigitsIs12 (n : ℕ) : Prop := sorry

-- Define a function to get the nth number in the sequence
def nthNumberWithSumOfDigits12 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem eleventhNumberWithSumOfDigits12Is156 : 
  nthNumberWithSumOfDigits12 11 = 156 := by sorry

end eleventhNumberWithSumOfDigits12Is156_l3256_325671


namespace base8_145_equals_101_in_base10_l3256_325639

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the base-8 number 145
def base8Number : List Nat := [5, 4, 1]

-- State the theorem
theorem base8_145_equals_101_in_base10 :
  base8ToBase10 base8Number = 101 := by
  sorry

end base8_145_equals_101_in_base10_l3256_325639


namespace largest_expression_l3256_325615

theorem largest_expression (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a + b) ≥ max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b)) := by
  sorry

end largest_expression_l3256_325615


namespace haunted_mansion_paths_l3256_325668

theorem haunted_mansion_paths (n : ℕ) (h : n = 8) : n * (n - 1) * (n - 2) = 336 := by
  sorry

end haunted_mansion_paths_l3256_325668


namespace rectangle_ratio_l3256_325675

/-- Given an arrangement of four congruent rectangles around an inner square,
    prove that the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) : 
  s > 0 →  -- inner square side length is positive
  x > 0 ∧ y > 0 →  -- rectangle sides are positive
  s + 2*y = 3*s →  -- outer square side length
  x + s = 3*s →  -- outer square side length (alternate direction)
  (3*s)^2 = 9*s^2 →  -- area relation
  x / y = 2 := by
  sorry

end rectangle_ratio_l3256_325675


namespace lunks_needed_for_twenty_apples_l3256_325667

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (4/7) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks needed to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (7/4) * kunks

theorem lunks_needed_for_twenty_apples : 
  lunks_for_apples 20 = 21 := by sorry

end lunks_needed_for_twenty_apples_l3256_325667


namespace sqrt_abs_equation_solution_l3256_325601

theorem sqrt_abs_equation_solution :
  ∀ x y : ℝ, Real.sqrt (2 * x + 3 * y) + |x + 3| = 0 → x = -3 ∧ y = 2 := by
  sorry

end sqrt_abs_equation_solution_l3256_325601


namespace vector_subtraction_scalar_multiplication_l3256_325665

theorem vector_subtraction_scalar_multiplication :
  (⟨3, -7⟩ : ℝ × ℝ) - 3 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-3, 5⟩ : ℝ × ℝ) := by
  sorry

end vector_subtraction_scalar_multiplication_l3256_325665


namespace hostel_provisions_theorem_l3256_325633

/-- The number of days provisions last for the initial group -/
def initial_days : ℕ := 50

/-- The number of days provisions last with 20 fewer people -/
def extended_days : ℕ := 250

/-- The number of fewer people in the extended scenario -/
def fewer_people : ℕ := 20

/-- The function to calculate the daily consumption rate given the number of people and days -/
def daily_consumption_rate (people : ℕ) (days : ℕ) : ℚ :=
  1 / (people.cast * days.cast)

theorem hostel_provisions_theorem (initial_girls : ℕ) :
  (daily_consumption_rate initial_girls initial_days =
   daily_consumption_rate (initial_girls + fewer_people) extended_days) →
  initial_girls = 25 := by
  sorry

end hostel_provisions_theorem_l3256_325633


namespace decreasing_function_property_l3256_325612

/-- A function f is decreasing on ℝ if for any x₁ < x₂, we have f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : DecreasingOn f) : f (a^2 + 1) < f a := by
  sorry

end decreasing_function_property_l3256_325612


namespace triangle_properties_l3256_325687

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  (Real.sin t.B = 4/5) ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → t.b^2 / (x^2 + y^2) ≥ 2/5) := by
  sorry

end triangle_properties_l3256_325687


namespace vector_problem_l3256_325674

theorem vector_problem (a b : ℝ × ℝ) :
  a + b = (2, 3) → a - b = (-2, 1) → a - 2 * b = (-4, 0) := by
  sorry

end vector_problem_l3256_325674


namespace junior_score_l3256_325640

theorem junior_score (n : ℝ) (h_n : n > 0) : 
  let junior_percent : ℝ := 0.2
  let senior_percent : ℝ := 0.8
  let total_average : ℝ := 85
  let senior_average : ℝ := 84
  let junior_score := (total_average * n - senior_average * senior_percent * n) / (junior_percent * n)
  junior_score = 89 := by sorry

end junior_score_l3256_325640


namespace tangent_line_is_correct_l3256_325625

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The point on the circle -/
def point_on_circle : ℝ × ℝ := (4, 1)

/-- The proposed tangent line equation -/
def tangent_line_equation (x y : ℝ) : Prop := 3*x + 4*y - 16 = 0 ∨ x = 4

/-- Theorem stating that the proposed equation represents the tangent line -/
theorem tangent_line_is_correct :
  tangent_line_equation (point_on_circle.1) (point_on_circle.2) ∧
  ∀ (x y : ℝ), circle_equation x y →
    tangent_line_equation x y →
    (x, y) = point_on_circle ∨
    ∃ (t : ℝ), (x, y) = (point_on_circle.1 + t, point_on_circle.2 + t) ∧ t ≠ 0 :=
sorry

end tangent_line_is_correct_l3256_325625


namespace restaurant_revenue_l3256_325636

/-- Calculates the total revenue from meals sold at a restaurant --/
theorem restaurant_revenue 
  (x y z : ℝ) -- Costs of kids, adult, and seniors' meals respectively
  (ratio_kids : ℕ) (ratio_adult : ℕ) (ratio_senior : ℕ) -- Ratio of meals sold
  (kids_meals_sold : ℕ) -- Number of kids meals sold
  (h_ratio : ratio_kids = 3 ∧ ratio_adult = 2 ∧ ratio_senior = 1) -- Given ratio
  (h_kids_sold : kids_meals_sold = 12) -- Given number of kids meals sold
  : 
  ∃ (total_revenue : ℝ),
    total_revenue = kids_meals_sold * x + 
      (kids_meals_sold * ratio_adult / ratio_kids) * y + 
      (kids_meals_sold * ratio_senior / ratio_kids) * z ∧
    total_revenue = 12 * x + 8 * y + 4 * z :=
by
  sorry

end restaurant_revenue_l3256_325636


namespace greg_additional_rotations_l3256_325689

/-- Represents the number of wheel rotations per block on flat ground. -/
def flatRotations : ℕ := 200

/-- Represents the number of wheel rotations per block uphill. -/
def uphillRotations : ℕ := 250

/-- Represents the number of blocks Greg has already ridden on flat ground. -/
def flatBlocksRidden : ℕ := 2

/-- Represents the number of blocks Greg has already ridden uphill. -/
def uphillBlocksRidden : ℕ := 1

/-- Represents the total number of wheel rotations Greg has already completed. -/
def rotationsCompleted : ℕ := 600

/-- Represents the number of additional uphill blocks Greg plans to ride. -/
def additionalUphillBlocks : ℕ := 3

/-- Represents the number of additional flat blocks Greg plans to ride. -/
def additionalFlatBlocks : ℕ := 2

/-- Represents the minimum total number of blocks Greg wants to ride. -/
def minTotalBlocks : ℕ := 8

/-- Theorem stating that Greg needs 550 more wheel rotations to reach his goal. -/
theorem greg_additional_rotations :
  let totalPlannedBlocks := flatBlocksRidden + uphillBlocksRidden + additionalFlatBlocks + additionalUphillBlocks
  let totalPlannedRotations := flatBlocksRidden * flatRotations + uphillBlocksRidden * uphillRotations +
                               additionalFlatBlocks * flatRotations + additionalUphillBlocks * uphillRotations
  totalPlannedBlocks ≥ minTotalBlocks ∧
  totalPlannedRotations - rotationsCompleted = 550 := by
  sorry


end greg_additional_rotations_l3256_325689


namespace cubic_unit_circle_roots_l3256_325670

theorem cubic_unit_circle_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∃ w₁ w₂ w₃ : ℂ, 
    (w₁^3 + Complex.abs a * w₁^2 + Complex.abs b * w₁ + Complex.abs c = 0) ∧
    (w₂^3 + Complex.abs a * w₂^2 + Complex.abs b * w₂ + Complex.abs c = 0) ∧
    (w₃^3 + Complex.abs a * w₃^2 + Complex.abs b * w₃ + Complex.abs c = 0) ∧
    Complex.abs w₁ = 1 ∧ Complex.abs w₂ = 1 ∧ Complex.abs w₃ = 1) :=
by sorry


end cubic_unit_circle_roots_l3256_325670


namespace tetrahedral_die_expected_steps_l3256_325676

def expected_steps (n : Nat) : ℚ :=
  match n with
  | 1 => 1
  | 2 => 5/4
  | 3 => 25/16
  | 4 => 125/64
  | _ => 0

theorem tetrahedral_die_expected_steps :
  let total_expectation := 1 + (expected_steps 1 + expected_steps 2 + expected_steps 3 + expected_steps 4) / 4
  total_expectation = 625/256 := by
  sorry

end tetrahedral_die_expected_steps_l3256_325676


namespace complex_quadrant_l3256_325600

theorem complex_quadrant (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l3256_325600


namespace square_and_product_l3256_325652

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : x = 42 ∧ (x + 2) * (x - 2) = 1760 := by
  sorry

end square_and_product_l3256_325652


namespace joey_age_when_beth_was_joeys_current_age_l3256_325602

/-- Represents a person's age at different points in time -/
structure Person where
  current_age : ℕ
  future_age : ℕ
  past_age : ℕ

/-- Given the conditions of the problem, prove that Joey was 4 years old when Beth was Joey's current age -/
theorem joey_age_when_beth_was_joeys_current_age 
  (joey : Person) 
  (beth : Person)
  (h1 : joey.current_age = 9)
  (h2 : joey.future_age = beth.current_age)
  (h3 : joey.future_age = joey.current_age + 5) :
  joey.past_age = 4 := by
sorry

end joey_age_when_beth_was_joeys_current_age_l3256_325602


namespace intersection_of_M_and_N_l3256_325690

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l3256_325690


namespace remainder_of_large_sum_l3256_325604

theorem remainder_of_large_sum (n : ℕ) : (7 * 10^20 + 2^20) % 11 = 9 :=
by sorry

end remainder_of_large_sum_l3256_325604


namespace dice_roll_probability_l3256_325606

/-- The probability of rolling an even number on a fair 6-sided die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number less than 3 on a fair 6-sided die -/
def prob_odd_lt_3 : ℚ := 1/6

/-- The number of ways to arrange two even numbers and one odd number -/
def num_arrangements : ℕ := 3

theorem dice_roll_probability :
  num_arrangements * (prob_even^2 * prob_odd_lt_3) = 1/8 := by
sorry

end dice_roll_probability_l3256_325606


namespace parallelogram_sides_l3256_325629

theorem parallelogram_sides (x y : ℝ) : 
  (2*x + 3 = 9) ∧ (8*y - 1 = 7) → x + y = 4 := by
  sorry

end parallelogram_sides_l3256_325629


namespace angle_ABH_measure_l3256_325684

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon in degrees. -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABH in a regular octagon ABCDEFGH in degrees. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := 22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5°. -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by sorry

end angle_ABH_measure_l3256_325684
