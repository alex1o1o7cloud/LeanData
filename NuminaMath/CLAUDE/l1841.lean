import Mathlib

namespace cards_difference_l1841_184118

theorem cards_difference (ann_cards : ℕ) (ann_heike_ratio : ℕ) (anton_heike_ratio : ℕ) :
  ann_cards = 60 →
  ann_heike_ratio = 6 →
  anton_heike_ratio = 3 →
  ann_cards - (anton_heike_ratio * (ann_cards / ann_heike_ratio)) = 30 := by
  sorry

end cards_difference_l1841_184118


namespace travel_time_difference_l1841_184194

/-- Given a set of 5 numbers (x, y, 10, 11, 9) with an average of 10 and a variance of 2, |x-y| = 4 -/
theorem travel_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 ∧ 
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry


end travel_time_difference_l1841_184194


namespace shiela_animal_drawings_l1841_184140

/-- Proves that each neighbor receives 8 animal drawings when Shiela distributes
    96 drawings equally among 12 neighbors. -/
theorem shiela_animal_drawings (neighbors : ℕ) (drawings : ℕ) (h1 : neighbors = 12) (h2 : drawings = 96) :
  drawings / neighbors = 8 := by
  sorry

end shiela_animal_drawings_l1841_184140


namespace andre_gave_23_flowers_l1841_184176

/-- The number of flowers Rosa initially had -/
def initial_flowers : ℕ := 67

/-- The number of flowers Rosa has now -/
def final_flowers : ℕ := 90

/-- The number of flowers Andre gave to Rosa -/
def andre_flowers : ℕ := final_flowers - initial_flowers

theorem andre_gave_23_flowers : andre_flowers = 23 := by
  sorry

end andre_gave_23_flowers_l1841_184176


namespace max_sum_xy_l1841_184123

def max_value_xy (x y : ℝ) : Prop :=
  (Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1) ∧
  ((x, y) ≠ (0, 0)) ∧
  (x^2 + y^2 ≠ 2)

theorem max_sum_xy :
  ∀ x y : ℝ, max_value_xy x y → x + y ≤ 1 + Real.sqrt 2 :=
by sorry

end max_sum_xy_l1841_184123


namespace m_value_l1841_184137

theorem m_value (m : ℝ) (M : Set ℝ) : M = {3, m + 1} → 4 ∈ M → m = 3 := by
  sorry

end m_value_l1841_184137


namespace exists_fib_divisible_by_2007_l1841_184131

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem exists_fib_divisible_by_2007 : ∃ n : ℕ, n > 0 ∧ 2007 ∣ fib n := by
  sorry

end exists_fib_divisible_by_2007_l1841_184131


namespace tims_garden_carrots_l1841_184116

/-- Represents the number of carrots in Tim's garden -/
def carrots : ℕ := sorry

/-- Represents the number of potatoes in Tim's garden -/
def potatoes : ℕ := sorry

/-- The ratio of carrots to potatoes -/
def ratio : Rat := 3 / 4

/-- The initial number of potatoes -/
def initial_potatoes : ℕ := 32

/-- The number of potatoes added -/
def added_potatoes : ℕ := 28

theorem tims_garden_carrots : 
  (ratio = carrots / potatoes) → 
  (potatoes = initial_potatoes + added_potatoes) →
  carrots = 45 := by sorry

end tims_garden_carrots_l1841_184116


namespace sum_of_cubes_nonnegative_l1841_184159

theorem sum_of_cubes_nonnegative (n : ℤ) (a b : ℚ) 
  (h1 : n > 1) 
  (h2 : n = a^3 + b^3) : 
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧ n = x^3 + y^3 := by
  sorry

end sum_of_cubes_nonnegative_l1841_184159


namespace polynomial_division_remainder_l1841_184196

theorem polynomial_division_remainder 
  (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 
  x^4 - 8*x^3 + 18*x^2 - 27*x + 15 = 
  (x^2 - 3*x + 14/3) * q x + (2*x + 205/9) :=
sorry

end polynomial_division_remainder_l1841_184196


namespace correct_average_weight_l1841_184111

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.9 := by
  sorry

end correct_average_weight_l1841_184111


namespace n_plus_one_in_terms_of_m_l1841_184168

theorem n_plus_one_in_terms_of_m (m n : ℕ) 
  (h1 : m * n = 121) 
  (h2 : (m + 1) * (n + 1) = 1000) : 
  n + 1 = 879 - m := by
sorry

end n_plus_one_in_terms_of_m_l1841_184168


namespace cooking_time_is_five_l1841_184158

def recommended_cooking_time (cooked_time seconds_remaining : ℕ) : ℚ :=
  (cooked_time + seconds_remaining) / 60

theorem cooking_time_is_five :
  recommended_cooking_time 45 255 = 5 := by
  sorry

end cooking_time_is_five_l1841_184158


namespace polygon_diagonals_l1841_184133

theorem polygon_diagonals (n : ℕ) : n ≥ 3 → (n - 3 = 5 ↔ n = 8) := by sorry

end polygon_diagonals_l1841_184133


namespace merchant_profit_percentage_l1841_184198

theorem merchant_profit_percentage
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (h_markup : markup_rate = 0.40)
  (h_discount : discount_rate = 0.15) :
  let marked_price := 1 + markup_rate
  let selling_price := marked_price * (1 - discount_rate)
  let profit_percentage := (selling_price - 1) * 100
  profit_percentage = 19 := by
sorry

end merchant_profit_percentage_l1841_184198


namespace intersection_point_l1841_184126

/-- Curve C₁ is defined by y = √x for x ≥ 0 -/
def C₁ (x y : ℝ) : Prop := y = Real.sqrt x ∧ x ≥ 0

/-- Curve C₂ is defined by x² + y² = 2 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- The point (1, 1) is the unique intersection point of curves C₁ and C₂ -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2) ∧ 
  (C₁ 1 1 ∧ C₂ 1 1) := by
  sorry

#check intersection_point

end intersection_point_l1841_184126


namespace max_sphere_radius_l1841_184187

-- Define the glass shape function
def glass_shape (x : ℝ) : ℝ := x^4

-- Define the circle equation
def circle_equation (x y r : ℝ) : Prop := x^2 + (y - r)^2 = r^2

-- Define the condition that the circle contains the origin
def contains_origin (r : ℝ) : Prop := circle_equation 0 0 r

-- Define the condition that the circle lies above or on the glass shape
def above_glass_shape (x y r : ℝ) : Prop := 
  circle_equation x y r → y ≥ glass_shape x

-- State the theorem
theorem max_sphere_radius : 
  ∃ (r : ℝ), r = (3 * 2^(1/3)) / 4 ∧ 
  (∀ (x y : ℝ), above_glass_shape x y r) ∧
  contains_origin r ∧
  (∀ (r' : ℝ), r' > r → ¬(∀ (x y : ℝ), above_glass_shape x y r') ∨ ¬(contains_origin r')) :=
sorry

end max_sphere_radius_l1841_184187


namespace error_probability_theorem_l1841_184167

-- Define the probability of error
def probability_of_error : ℝ := 0.01

-- Define the observed value of K²
def observed_k_squared : ℝ := 6.635

-- Define the relationship between variables
def relationship_exists : Prop := True

-- Define the conclusion of the statistical test
def statistical_conclusion (p : ℝ) (relationship : Prop) : Prop :=
  p ≤ probability_of_error ∧ relationship

-- Theorem statement
theorem error_probability_theorem 
  (h : statistical_conclusion probability_of_error relationship_exists) :
  probability_of_error = 0.01 := by sorry

end error_probability_theorem_l1841_184167


namespace algebraic_expression_value_l1841_184180

theorem algebraic_expression_value (m n : ℝ) (h : -2*m + 3*n^2 = -7) : 
  12*n^2 - 8*m + 4 = -24 := by sorry

end algebraic_expression_value_l1841_184180


namespace cube_opposite_faces_l1841_184122

/-- Represents a face of a cube --/
inductive Face : Type
| G | H | I | J | S | K

/-- Represents the adjacency relation between faces --/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces --/
def opposite : Face → Face → Prop := sorry

/-- Theorem: If H and I are adjacent, G is adjacent to both H and I, 
    and J is adjacent to H and I, then J is opposite to G --/
theorem cube_opposite_faces 
  (adj_H_I : adjacent Face.H Face.I)
  (adj_G_H : adjacent Face.G Face.H)
  (adj_G_I : adjacent Face.G Face.I)
  (adj_J_H : adjacent Face.J Face.H)
  (adj_J_I : adjacent Face.J Face.I) :
  opposite Face.G Face.J := by sorry

end cube_opposite_faces_l1841_184122


namespace projection_of_sum_onto_a_l1841_184107

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-2, 1)

theorem projection_of_sum_onto_a :
  let sum := (a.1 + b.1, a.2 + b.2)
  let dot_product := sum.1 * a.1 + sum.2 * a.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product / magnitude_a = Real.sqrt 2 / 2 := by sorry

end projection_of_sum_onto_a_l1841_184107


namespace polynomial_factorization_l1841_184112

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x + 63) * (x + 3) * (x + 5) * (x + 21) := by
sorry

end polynomial_factorization_l1841_184112


namespace fifth_element_row_20_value_l1841_184110

/-- Pascal's triangle element -/
def pascal_triangle_element (n k : ℕ) : ℕ := Nat.choose n k

/-- The fifth element in Row 20 of Pascal's triangle -/
def fifth_element_row_20 : ℕ := pascal_triangle_element 20 4

theorem fifth_element_row_20_value : fifth_element_row_20 = 4845 := by
  sorry

end fifth_element_row_20_value_l1841_184110


namespace average_of_remaining_numbers_l1841_184179

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℝ)
  (group1 : ℕ)
  (avg1 : ℝ)
  (group2 : ℕ)
  (avg2 : ℝ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_group1 : group1 = 2)
  (h_avg1 : avg1 = 3.4)
  (h_group2 : group2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let remaining := total - group1 - group2
  let sum_all := total * avg_all
  let sum1 := group1 * avg1
  let sum2 := group2 * avg2
  let sum_remaining := sum_all - sum1 - sum2
  sum_remaining / remaining = 4.6 := by sorry

end average_of_remaining_numbers_l1841_184179


namespace impossible_non_eleven_multiple_l1841_184108

/-- Represents a 5x5 board where each cell can be increased along with its adjacent cells. -/
structure Board :=
  (cells : Matrix (Fin 5) (Fin 5) ℕ)

/-- The operation of increasing a cell and its adjacent cells by 1. -/
def increase_cell (b : Board) (i j : Fin 5) : Board := sorry

/-- Checks if all cells in the board have the same value. -/
def all_cells_equal (b : Board) (s : ℕ) : Prop := sorry

/-- Main theorem: It's impossible to obtain a number not divisible by 11 in all cells. -/
theorem impossible_non_eleven_multiple (s : ℕ) (h : ¬ 11 ∣ s) : 
  ¬ ∃ (b : Board), all_cells_equal b s :=
sorry

end impossible_non_eleven_multiple_l1841_184108


namespace range_of_a_l1841_184135

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*x + 3*a > 0) → a > 1 := by
  sorry

end range_of_a_l1841_184135


namespace house_sale_tax_percentage_l1841_184124

theorem house_sale_tax_percentage (market_value : ℝ) (over_market_percentage : ℝ) 
  (num_people : ℕ) (amount_per_person : ℝ) :
  market_value = 500000 →
  over_market_percentage = 0.20 →
  num_people = 4 →
  amount_per_person = 135000 →
  (market_value * (1 + over_market_percentage) - num_people * amount_per_person) / 
    (market_value * (1 + over_market_percentage)) = 0.10 := by
  sorry

end house_sale_tax_percentage_l1841_184124


namespace tan_alpha_3_implies_fraction_l1841_184160

theorem tan_alpha_3_implies_fraction (α : Real) (h : Real.tan α = 3) :
  (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 := by
  sorry

end tan_alpha_3_implies_fraction_l1841_184160


namespace pen_price_calculation_l1841_184151

/-- Given the purchase of pens and pencils with known quantities and prices,
    prove that the average price of a pen is $14.00. -/
theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) 
    (pencil_price : ℚ) (pen_price : ℚ) : 
    num_pens = 30 → 
    num_pencils = 75 → 
    total_cost = 570 → 
    pencil_price = 2 → 
    pen_price = (total_cost - num_pencils * pencil_price) / num_pens → 
    pen_price = 14 := by
  sorry

end pen_price_calculation_l1841_184151


namespace seminar_ratio_l1841_184155

theorem seminar_ratio (total_attendees : ℕ) (avg_age_all : ℚ) (avg_age_doctors : ℚ) (avg_age_lawyers : ℚ)
  (h_total : total_attendees = 20)
  (h_avg_all : avg_age_all = 45)
  (h_avg_doctors : avg_age_doctors = 40)
  (h_avg_lawyers : avg_age_lawyers = 55) :
  ∃ (num_doctors num_lawyers : ℚ),
    num_doctors + num_lawyers = total_attendees ∧
    (num_doctors * avg_age_doctors + num_lawyers * avg_age_lawyers) / total_attendees = avg_age_all ∧
    num_doctors / num_lawyers = 2 := by
  sorry


end seminar_ratio_l1841_184155


namespace johns_next_birthday_age_l1841_184104

/-- Proves that John's age on his next birthday is 9, given the conditions of the problem -/
theorem johns_next_birthday_age (john carl beth : ℝ) 
  (h1 : john = 0.75 * carl)  -- John is 25% younger than Carl
  (h2 : carl = 1.3 * beth)   -- Carl is 30% older than Beth
  (h3 : john + carl + beth = 30) -- Sum of their ages is 30
  : ⌈john⌉ = 9 := by
  sorry

end johns_next_birthday_age_l1841_184104


namespace absolute_value_squared_l1841_184199

theorem absolute_value_squared (a b : ℝ) : |a| > |b| → a^2 > b^2 := by
  sorry

end absolute_value_squared_l1841_184199


namespace max_sum_with_lcm_gcd_constraint_l1841_184169

theorem max_sum_with_lcm_gcd_constraint (m n : ℕ) : 
  m + 3*n - 5 = 2*(Nat.lcm m n) - 11*(Nat.gcd m n) → 
  m + n ≤ 70 ∧ ∃ (m₀ n₀ : ℕ), m₀ + 3*n₀ - 5 = 2*(Nat.lcm m₀ n₀) - 11*(Nat.gcd m₀ n₀) ∧ m₀ + n₀ = 70 :=
by sorry

end max_sum_with_lcm_gcd_constraint_l1841_184169


namespace slacks_percentage_is_25_percent_l1841_184136

/-- Represents the clothing items and their quantities -/
structure Wardrobe where
  blouses : ℕ
  skirts : ℕ
  slacks : ℕ

/-- Represents the percentages of clothing items in the hamper -/
structure HamperPercentages where
  blouses : ℚ
  skirts : ℚ
  slacks : ℚ

/-- Calculates the percentage of slacks in the hamper -/
def calculate_slacks_percentage (w : Wardrobe) (h : HamperPercentages) (total_in_washer : ℕ) : ℚ :=
  let blouses_in_hamper := (w.blouses : ℚ) * h.blouses
  let skirts_in_hamper := (w.skirts : ℚ) * h.skirts
  let slacks_in_hamper := (total_in_washer : ℚ) - blouses_in_hamper - skirts_in_hamper
  slacks_in_hamper / (w.slacks : ℚ)

/-- Theorem stating that the percentage of slacks in the hamper is 25% -/
theorem slacks_percentage_is_25_percent (w : Wardrobe) (h : HamperPercentages) :
  w.blouses = 12 →
  w.skirts = 6 →
  w.slacks = 8 →
  h.blouses = 3/4 →
  h.skirts = 1/2 →
  calculate_slacks_percentage w h 14 = 1/4 := by
  sorry

#eval (1 : ℚ) / 4  -- To verify that 1/4 is indeed 25%

end slacks_percentage_is_25_percent_l1841_184136


namespace ratio_comparison_l1841_184157

theorem ratio_comparison (a : ℚ) (h : a > 3) : (3 : ℚ) / 4 < a / 4 := by
  sorry

end ratio_comparison_l1841_184157


namespace decimal_comparisons_l1841_184193

theorem decimal_comparisons : 
  (3 > 2.95) ∧ (0.08 < 0.21) ∧ (0.6 = 0.60) := by
  sorry

end decimal_comparisons_l1841_184193


namespace quadratic_trinomial_decomposition_l1841_184129

theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4*p*r = 0) ∧
    (t^2 - 4*s*u = 0) :=
by sorry

end quadratic_trinomial_decomposition_l1841_184129


namespace isabellas_paintable_area_l1841_184142

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable area for Isabella's bedrooms is 1552 square feet. -/
theorem isabellas_paintable_area :
  total_paintable_area 4 14 12 9 80 = 1552 := by
  sorry

end isabellas_paintable_area_l1841_184142


namespace police_catches_thief_l1841_184127

-- Define the square courtyard
def side_length : ℝ := 340

-- Define speeds
def police_speed : ℝ := 85
def thief_speed : ℝ := 75

-- Define the time to catch
def time_to_catch : ℝ := 44

-- Theorem statement
theorem police_catches_thief :
  let time_to_sight : ℝ := (4 * side_length) / (police_speed - thief_speed)
  let police_distance : ℝ := police_speed * time_to_sight
  let thief_distance : ℝ := thief_speed * time_to_sight
  let remaining_side : ℝ := side_length - (thief_distance % side_length)
  let chase_time : ℝ := Real.sqrt ((remaining_side^2) / (police_speed^2 - thief_speed^2))
  time_to_sight + chase_time = time_to_catch :=
by sorry

end police_catches_thief_l1841_184127


namespace concatenated_digits_2015_l1841_184164

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The sum of digits for all numbers from 1 to n -/
def sum_digits (n : ℕ) : ℕ := sorry

theorem concatenated_digits_2015 : sum_digits 2015 = 6953 := by sorry

end concatenated_digits_2015_l1841_184164


namespace journey_distance_l1841_184106

/-- The total distance of a journey is the sum of miles driven and miles remaining. -/
theorem journey_distance (miles_driven miles_remaining : ℕ) 
  (h1 : miles_driven = 923)
  (h2 : miles_remaining = 277) :
  miles_driven + miles_remaining = 1200 := by
sorry

end journey_distance_l1841_184106


namespace complex_equation_product_l1841_184100

theorem complex_equation_product (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a + b * i = 5 / (1 + 2 * i) →
  a * b = -2 := by
  sorry

end complex_equation_product_l1841_184100


namespace teal_color_survey_l1841_184148

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_blue = 85)
  (h3 : both = 47)
  (h4 : neither = 22) :
  total - (more_blue - both + both + neither) = 90 := by
  sorry

end teal_color_survey_l1841_184148


namespace biology_quiz_probability_l1841_184182

theorem biology_quiz_probability : 
  let n : ℕ := 6  -- number of guessed questions
  let k : ℕ := 4  -- number of possible answers per question
  let p : ℚ := 1 / k  -- probability of guessing correctly on a single question
  1 - (1 - p) ^ n = 3367 / 4096 :=
by
  sorry

end biology_quiz_probability_l1841_184182


namespace sum_of_specific_numbers_l1841_184115

theorem sum_of_specific_numbers : 
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := by
  sorry

end sum_of_specific_numbers_l1841_184115


namespace intersection_with_complement_l1841_184165

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4, 5}
def N : Finset Nat := {1, 3}

theorem intersection_with_complement : M ∩ (U \ N) = {4, 5} := by sorry

end intersection_with_complement_l1841_184165


namespace wax_sculpture_theorem_l1841_184125

/-- Proves that the total number of wax sticks used is 20 --/
theorem wax_sculpture_theorem (large_sticks small_sticks : ℕ) 
  (h1 : large_sticks = 4)
  (h2 : small_sticks = 2)
  (small_animals large_animals : ℕ)
  (h3 : small_animals = 3 * large_animals)
  (total_small_sticks : ℕ)
  (h4 : total_small_sticks = 12)
  (h5 : total_small_sticks = small_animals * small_sticks) :
  total_small_sticks + large_animals * large_sticks = 20 := by
sorry

end wax_sculpture_theorem_l1841_184125


namespace min_value_triangle_sides_l1841_184153

/-- 
Given a triangle with side lengths x+10, x+5, and 4x, where the angle opposite to side 4x
is the largest angle, the minimum value of 4x - (x+5) is 5.
-/
theorem min_value_triangle_sides (x : ℝ) : 
  (x + 5 + 4*x > x + 10) ∧ 
  (x + 5 + x + 10 > 4*x) ∧ 
  (4*x + x + 10 > x + 5) ∧
  (4*x > x + 5) ∧ 
  (4*x > x + 10) →
  ∃ (y : ℝ), y ≥ x ∧ ∀ (z : ℝ), z ≥ x → 4*z - (z + 5) ≥ 4*y - (y + 5) ∧ 4*y - (y + 5) = 5 := by
  sorry

end min_value_triangle_sides_l1841_184153


namespace carrie_fourth_day_miles_l1841_184143

/-- Represents Carrie's four-day trip --/
structure CarrieTrip where
  day1_miles : ℕ
  day2_miles : ℕ
  day3_miles : ℕ
  day4_miles : ℕ
  charge_interval : ℕ
  total_charges : ℕ

/-- Theorem: Given the conditions of Carrie's trip, she drove 189 miles on the fourth day --/
theorem carrie_fourth_day_miles (trip : CarrieTrip)
  (h1 : trip.day1_miles = 135)
  (h2 : trip.day2_miles = trip.day1_miles + 124)
  (h3 : trip.day3_miles = 159)
  (h4 : trip.charge_interval = 106)
  (h5 : trip.total_charges = 7)
  : trip.day4_miles = 189 := by
  sorry

#check carrie_fourth_day_miles

end carrie_fourth_day_miles_l1841_184143


namespace expected_ones_is_one_third_l1841_184184

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The expected number of 1's when rolling two standard dice -/
def expected_ones : ℚ := 2 * (prob_one * prob_one) + 1 * (2 * prob_one * prob_not_one)

theorem expected_ones_is_one_third : expected_ones = 1 / 3 := by
  sorry

end expected_ones_is_one_third_l1841_184184


namespace phil_final_quarters_l1841_184101

/-- Calculates the number of quarters Phil has after four years of collecting and losing some. -/
def phil_quarters : ℕ :=
  let initial := 50
  let after_first_year := initial * 2
  let second_year_collection := 3 * 12
  let third_year_collection := 12 / 3
  let total_before_loss := after_first_year + second_year_collection + third_year_collection
  let quarters_lost := total_before_loss / 4
  total_before_loss - quarters_lost

/-- Theorem stating that Phil ends up with 105 quarters after four years. -/
theorem phil_final_quarters : phil_quarters = 105 := by
  sorry

end phil_final_quarters_l1841_184101


namespace hyperbola_iff_equation_l1841_184109

/-- Represents the condition for a hyperbola given a real number m -/
def is_hyperbola (m : ℝ) : Prop :=
  (m < -1) ∨ (-1 < m ∧ m < 1) ∨ (m > 2)

/-- The equation representing a potential hyperbola -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

/-- Theorem stating the equivalence between the hyperbola condition and the equation -/
theorem hyperbola_iff_equation (m : ℝ) :
  is_hyperbola m ↔ ∃ x y : ℝ, hyperbola_equation m x y :=
sorry

end hyperbola_iff_equation_l1841_184109


namespace fraction_equality_l1841_184105

theorem fraction_equality (x : ℚ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) ↔ x = -1/2 := by
  sorry

end fraction_equality_l1841_184105


namespace power_fraction_simplification_l1841_184178

theorem power_fraction_simplification :
  ((2^5) * (9^2)) / ((8^2) * (3^5)) = 1/6 := by sorry

end power_fraction_simplification_l1841_184178


namespace arithmetic_sequence_problem_l1841_184171

/-- Given an arithmetic sequence {1/aₙ} where a₁ = 1 and a₄ = 4, prove that a₁₀ = -4/5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∃ d : ℚ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d) →
  a 1 = 1 →
  a 4 = 4 →
  a 10 = -4/5 := by
sorry

end arithmetic_sequence_problem_l1841_184171


namespace gcd_of_factorials_l1841_184138

theorem gcd_of_factorials : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_of_factorials_l1841_184138


namespace find_number_l1841_184141

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 63 :=
  sorry

end find_number_l1841_184141


namespace horner_method_v₃_l1841_184174

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 3*x + 2

def horner_v₃ (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v₀ := 1
  let v₁ := x + (-5)
  let v₂ := v₁ * x + 6
  v₂ * x + 0

theorem horner_method_v₃ :
  horner_v₃ f (-2) = -40 := by sorry

end horner_method_v₃_l1841_184174


namespace lawn_mowing_time_l1841_184191

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time (lawn_length lawn_width swath_width overlap mowing_rate : ℝ) :
  lawn_length = 120 →
  lawn_width = 180 →
  swath_width = 30 / 12 →
  overlap = 6 / 12 →
  mowing_rate = 4000 →
  (lawn_width / (swath_width - overlap) * lawn_length) / mowing_rate = 2.7 := by
  sorry

end lawn_mowing_time_l1841_184191


namespace tangent_line_intersection_l1841_184147

theorem tangent_line_intersection (x₀ : ℝ) (m : ℝ) : 
  (0 < m) → (m < 1) →
  (2 * x₀ = 1 / m) →
  (x₀^2 - Real.log (2 * x₀) - 1 = 0) →
  (Real.sqrt 2 < x₀) ∧ (x₀ < Real.sqrt 3) := by
  sorry

end tangent_line_intersection_l1841_184147


namespace square_root_expression_equals_256_l1841_184170

theorem square_root_expression_equals_256 :
  Real.sqrt ((16^12 + 2^36) / (16^5 + 2^42)) = 256 := by
  sorry

end square_root_expression_equals_256_l1841_184170


namespace max_area_rectangle_fixed_perimeter_l1841_184166

/-- The maximum area of a rectangle with perimeter 30 meters is 56.25 square meters. -/
theorem max_area_rectangle_fixed_perimeter :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 30 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * (l' + w') = 30 → l' * w' ≤ l * w) ∧
  l * w = 56.25 := by
  sorry

end max_area_rectangle_fixed_perimeter_l1841_184166


namespace pizza_slices_count_l1841_184175

-- Define the number of pizzas
def num_pizzas : Nat := 4

-- Define the number of slices for each type of pizza
def slices_first_two : Nat := 8
def slices_third : Nat := 10
def slices_fourth : Nat := 12

-- Define the total number of slices
def total_slices : Nat := 2 * slices_first_two + slices_third + slices_fourth

-- Theorem to prove
theorem pizza_slices_count : total_slices = 38 := by
  sorry

end pizza_slices_count_l1841_184175


namespace ticket_cost_count_l1841_184102

def ticket_cost_possibilities (total_11th : ℕ) (total_12th : ℕ) : ℕ :=
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ 
    total_11th % x = 0 ∧ 
    total_12th % x = 0 ∧ 
    total_11th / x < total_12th / x) 
    (Finset.range (min total_11th total_12th + 1))).card

theorem ticket_cost_count : ticket_cost_possibilities 108 90 = 6 := by
  sorry

end ticket_cost_count_l1841_184102


namespace circle_diameter_from_area_l1841_184163

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_from_area_l1841_184163


namespace average_price_per_book_l1841_184195

-- Define the problem parameters
def books_shop1 : ℕ := 32
def cost_shop1 : ℕ := 1500
def books_shop2 : ℕ := 60
def cost_shop2 : ℕ := 340

-- Theorem to prove
theorem average_price_per_book :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 20 := by
  sorry


end average_price_per_book_l1841_184195


namespace arithmetic_mean_problem_l1841_184177

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
  sorry

end arithmetic_mean_problem_l1841_184177


namespace largest_and_smallest_A_l1841_184161

/-- Given a nine-digit number B, returns the number A obtained by moving the last digit of B to the first place -/
def getA (B : ℕ) : ℕ :=
  (B % 10) * 10^8 + B / 10

/-- Checks if two natural numbers are coprime -/
def isCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem largest_and_smallest_A :
  ∃ (A_max A_min : ℕ),
    (∀ A B : ℕ,
      B > 22222222 →
      isCoprime B 18 →
      A = getA B →
      A ≤ A_max ∧ A ≥ A_min) ∧
    A_max = 999999998 ∧
    A_min = 122222224 := by
  sorry

end largest_and_smallest_A_l1841_184161


namespace fixed_point_on_all_lines_fixed_point_unique_l1841_184139

/-- The fixed point through which all lines of the form ax + y + 1 = 0 pass -/
def fixed_point : ℝ × ℝ := (0, -1)

/-- The equation of the line ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

theorem fixed_point_on_all_lines :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ a : ℝ, line_equation a x y) → (x, y) = fixed_point :=
by sorry

end fixed_point_on_all_lines_fixed_point_unique_l1841_184139


namespace r_value_when_n_is_3_l1841_184189

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + 1
  let r : ℕ := 3^s - s + 2
  r = 19676 := by sorry

end r_value_when_n_is_3_l1841_184189


namespace expression_evaluation_l1841_184146

theorem expression_evaluation : 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8) = -50 := by
  sorry

end expression_evaluation_l1841_184146


namespace count_integer_ratios_eq_five_l1841_184156

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  sorry

/-- Given two arithmetic sequences and a property of their sums,
    counts the number of positive integers that make the ratio of their terms an integer -/
def count_integer_ratios (a b : ArithmeticSequence) : ℕ :=
  sorry

theorem count_integer_ratios_eq_five
  (a b : ArithmeticSequence)
  (h : ∀ n : ℕ+, sum_n a n / sum_n b n = (7 * n + 45) / (n + 3)) :
  count_integer_ratios a b = 5 :=
sorry

end count_integer_ratios_eq_five_l1841_184156


namespace parabola_directrix_l1841_184172

/-- The directrix of the parabola y = (x^2 - 8x + 12) / 16 is y = -17/64 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 8*x + 12) / 16
  ∃ (a b c : ℝ), (∀ x, f x = a * (x - b)^2 + c) ∧
                 (a ≠ 0) ∧
                 (c - 1 / (4 * a) = -17/64) :=
sorry

end parabola_directrix_l1841_184172


namespace fraction_difference_l1841_184173

theorem fraction_difference (a b : ℝ) (h : b / a = 2) :
  b / (a + b) - a / (a + b) = 1 / 3 := by sorry

end fraction_difference_l1841_184173


namespace camping_trip_percentage_l1841_184152

theorem camping_trip_percentage (total_students : ℝ) (students_over_100 : ℝ) (students_100_or_less : ℝ) :
  students_over_100 = 0.16 * total_students →
  students_over_100 + students_100_or_less = 0.64 * total_students →
  (students_over_100 + students_100_or_less) / total_students = 0.64 :=
by
  sorry

end camping_trip_percentage_l1841_184152


namespace solve_cube_equation_l1841_184120

theorem solve_cube_equation : ∃ x : ℝ, (x - 3)^3 = (1/27)⁻¹ ∧ x = 6 := by
  sorry

end solve_cube_equation_l1841_184120


namespace function_inequality_l1841_184121

theorem function_inequality (a x : ℝ) : 
  let f := fun (t : ℝ) => t^2 - t + 13
  |x - a| < 1 → |f x - f a| < 2 * (|a| + 1) := by
sorry

end function_inequality_l1841_184121


namespace factory_defect_rate_l1841_184183

theorem factory_defect_rate (total_output : ℝ) : 
  let machine_a_output := 0.4 * total_output
  let machine_b_output := 0.6 * total_output
  let machine_a_defect_rate := 9 / 1000
  let total_defect_rate := 0.0156
  ∃ (machine_b_defect_rate : ℝ),
    0.4 * machine_a_defect_rate + 0.6 * machine_b_defect_rate = total_defect_rate ∧
    1 / machine_b_defect_rate = 50 := by
  sorry

end factory_defect_rate_l1841_184183


namespace equations_hold_l1841_184103

-- Define the equations
def equation1 : ℝ := 6.8 + 4.1 + 1.1
def equation2 : ℝ := 6.2 + 6.2 + 7.6
def equation3 : ℝ := 19.9 - 4.3 - 5.6

-- State the theorem
theorem equations_hold :
  equation1 = 12 ∧ equation2 = 20 ∧ equation3 = 10 := by sorry

end equations_hold_l1841_184103


namespace no_common_points_l1841_184185

theorem no_common_points : 
  ¬∃ (x y : ℝ), (x^2 + y^2 = 4) ∧ (x^2 + 2*y^2 = 2) := by
  sorry

end no_common_points_l1841_184185


namespace unique_solution_linear_system_l1841_184188

theorem unique_solution_linear_system
  (a b c d : ℝ)
  (h : a * d - c * b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y = 0 ∧ c * x + d * y = 0 → x = 0 ∧ y = 0 :=
by sorry

end unique_solution_linear_system_l1841_184188


namespace chord_length_l1841_184117

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B → abs (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end chord_length_l1841_184117


namespace crease_length_l1841_184197

theorem crease_length (width : Real) (θ : Real) : width = 10 → 
  let crease_length := width / 2 * Real.tan θ
  crease_length = 5 * Real.tan θ := by sorry

end crease_length_l1841_184197


namespace negation_equivalence_l1841_184113

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, (x₀^2 + 1 > 0 ∨ x₀ > Real.sin x₀)) ↔ 
  (∀ x : ℝ, (x^2 + 1 ≤ 0 ∧ x ≤ Real.sin x)) :=
by sorry

end negation_equivalence_l1841_184113


namespace rectangle_to_total_height_ratio_l1841_184128

/-- Represents an octagon with specific properties -/
structure Octagon :=
  (area : ℝ)
  (rectangle_width : ℝ)
  (triangle_base : ℝ)

/-- Properties of the octagon -/
axiom octagon_properties (o : Octagon) :
  o.area = 12 ∧
  o.rectangle_width = 3 ∧
  o.triangle_base = 3

/-- The diagonal bisects the area of the octagon -/
axiom diagonal_bisects (o : Octagon) (rectangle_height : ℝ) :
  o.rectangle_width * rectangle_height = o.area / 2

/-- The total height of the octagon -/
def total_height (o : Octagon) (rectangle_height : ℝ) : ℝ :=
  2 * rectangle_height

/-- Theorem: The ratio of rectangle height to total height is 1/2 -/
theorem rectangle_to_total_height_ratio (o : Octagon) (rectangle_height : ℝ) :
  rectangle_height / (total_height o rectangle_height) = 1 / 2 := by
  sorry

end rectangle_to_total_height_ratio_l1841_184128


namespace A_subset_B_A_eq_B_when_single_element_l1841_184150

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem 1: A ⊆ B
theorem A_subset_B (a b : ℝ) : A a b ⊆ B a b := by sorry

-- Theorem 2: If A has only one element, then A = B
theorem A_eq_B_when_single_element (a b : ℝ) :
  (∃! x, x ∈ A a b) → A a b = B a b := by sorry

end A_subset_B_A_eq_B_when_single_element_l1841_184150


namespace ed_lighter_than_al_l1841_184149

/-- Prove that Ed is 38 pounds lighter than Al given the following conditions:
  * Al is 25 pounds heavier than Ben
  * Ben is 16 pounds lighter than Carl
  * Ed weighs 146 pounds
  * Carl weighs 175 pounds
-/
theorem ed_lighter_than_al (carl_weight ben_weight al_weight ed_weight : ℕ) : 
  carl_weight = 175 →
  ben_weight = carl_weight - 16 →
  al_weight = ben_weight + 25 →
  ed_weight = 146 →
  al_weight - ed_weight = 38 := by
  sorry

#check ed_lighter_than_al

end ed_lighter_than_al_l1841_184149


namespace circle_radius_is_6_sqrt_2_l1841_184132

/-- A right triangle with squares constructed on two sides --/
structure RightTriangleWithSquares where
  -- The lengths of the two sides of the right triangle
  PQ : ℝ
  QR : ℝ
  -- Assertion that the squares are constructed on these sides
  square_PQ_constructed : Bool
  square_QR_constructed : Bool
  -- Assertion that the corners of the squares lie on a circle
  corners_on_circle : Bool

/-- The radius of the circle passing through the corners of the squares --/
def circle_radius (t : RightTriangleWithSquares) : ℝ :=
  sorry

/-- Theorem stating that for a right triangle with PQ = 9 and QR = 12,
    and squares constructed on these sides, if the corners of the squares
    lie on a circle, then the radius of this circle is 6√2 --/
theorem circle_radius_is_6_sqrt_2 (t : RightTriangleWithSquares)
    (h1 : t.PQ = 9)
    (h2 : t.QR = 12)
    (h3 : t.square_PQ_constructed = true)
    (h4 : t.square_QR_constructed = true)
    (h5 : t.corners_on_circle = true) :
  circle_radius t = 6 * Real.sqrt 2 :=
by sorry

end circle_radius_is_6_sqrt_2_l1841_184132


namespace sum_leq_fourth_powers_over_product_l1841_184192

theorem sum_leq_fourth_powers_over_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) := by
  sorry

end sum_leq_fourth_powers_over_product_l1841_184192


namespace polynomial_factorization_l1841_184144

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 2)^3 := by
  sorry

end polynomial_factorization_l1841_184144


namespace optimal_split_positions_l1841_184190

/-- The number N as defined in the problem -/
def N : ℕ := 10^1001 - 1

/-- Function to calculate the sum when splitting at position m -/
def S (m : ℕ) : ℕ := 2 * 10^m + 10^(1992 - m) - 10

/-- Function to calculate the product when splitting at position m -/
def P (m : ℕ) : ℕ := 2 * 10^1992 + 9 - 18 * 10^m - 10^(1992 - m)

/-- Theorem stating the optimal split positions for sum and product -/
theorem optimal_split_positions :
  (∀ m, m ≠ 996 → S 996 ≤ S m) ∧
  (∀ m, m ≠ 995 → P 995 ≥ P m) :=
sorry


end optimal_split_positions_l1841_184190


namespace always_ahead_probability_l1841_184114

/-- Represents the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process 
    in an election where A receives n votes and B receives m votes. -/
def election_probability (n m : ℕ) : ℚ :=
  (n - m : ℚ) / (n + m : ℚ)

/-- Theorem stating the probability that candidate A's cumulative vote count 
    always remains ahead of candidate B's during the counting process. -/
theorem always_ahead_probability (n m : ℕ) (h : n > m) :
  election_probability n m = (n - m : ℚ) / (n + m : ℚ) := by
  sorry

end always_ahead_probability_l1841_184114


namespace m_greater_than_n_l1841_184130

theorem m_greater_than_n : ∀ a : ℝ, 2 * a^2 - 4 * a > a^2 - 2 * a - 3 := by
  sorry

end m_greater_than_n_l1841_184130


namespace beanie_babies_per_stocking_l1841_184145

theorem beanie_babies_per_stocking : 
  ∀ (candy_canes_per_stocking : ℕ) 
    (books_per_stocking : ℕ) 
    (num_stockings : ℕ) 
    (total_stuffers : ℕ),
  candy_canes_per_stocking = 4 →
  books_per_stocking = 1 →
  num_stockings = 3 →
  total_stuffers = 21 →
  (total_stuffers - (candy_canes_per_stocking + books_per_stocking) * num_stockings) / num_stockings = 2 :=
by
  sorry

end beanie_babies_per_stocking_l1841_184145


namespace solution_existence_l1841_184134

/-- Given a positive real number a, prove the existence of real solutions for two systems of equations involving parameter m. -/
theorem solution_existence (a : ℝ) (ha : a > 0) :
  (∀ m : ℝ, ∃ x y : ℝ, y = m * x + a ∧ 1 / x - 1 / y = 1 / a) ∧
  (∀ m : ℝ, (m ≤ 0 ∨ m ≥ 4) → ∃ x y : ℝ, y = m * x - a ∧ 1 / x - 1 / y = 1 / a) :=
by sorry

end solution_existence_l1841_184134


namespace polynomial_division_theorem_l1841_184181

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^4 - 5*x^3 + 9 = 
  (x - 2) * (x^5 + 2*x^4 + 6*x^3 + 7*x^2 + 14*x + 28) + R :=
by sorry

end polynomial_division_theorem_l1841_184181


namespace x_power_4095_minus_reciprocal_l1841_184154

theorem x_power_4095_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^4095 - 1/x^4095 = 20 * Real.sqrt 2 := by
  sorry

end x_power_4095_minus_reciprocal_l1841_184154


namespace square_sum_equality_l1841_184186

theorem square_sum_equality (x y : ℝ) (h : x + y = -2) : x^2 + y^2 + 2*x*y = 4 := by
  sorry

end square_sum_equality_l1841_184186


namespace coat_price_calculation_l1841_184119

def calculate_final_price (initial_price : ℝ) (initial_tax_rate : ℝ) 
                          (discount_rate : ℝ) (additional_discount : ℝ) 
                          (final_tax_rate : ℝ) : ℝ :=
  let price_after_initial_tax := initial_price * (1 + initial_tax_rate)
  let price_after_discount := price_after_initial_tax * (1 - discount_rate)
  let price_after_additional_discount := price_after_discount - additional_discount
  price_after_additional_discount * (1 + final_tax_rate)

theorem coat_price_calculation :
  calculate_final_price 200 0.10 0.25 10 0.05 = 162.75 := by
  sorry

end coat_price_calculation_l1841_184119


namespace billion_to_scientific_notation_l1841_184162

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation 1673000000 = ScientificNotation.mk 1.673 9 (by norm_num) :=
sorry

end billion_to_scientific_notation_l1841_184162
