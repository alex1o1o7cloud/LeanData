import Mathlib

namespace NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l56_5608

theorem sphere_volume_from_cube_surface (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 3 → 
  (6 * cube_side^2 : ℝ) = 4 * π * sphere_radius^2 → 
  (4 / 3 : ℝ) * π * sphere_radius^3 = 54 * Real.sqrt 6 / Real.sqrt π := by
  sorry

#check sphere_volume_from_cube_surface

end NUMINAMATH_CALUDE_sphere_volume_from_cube_surface_l56_5608


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l56_5627

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l56_5627


namespace NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l56_5656

theorem fraction_equality_implies_x_value :
  ∀ x : ℝ, (4 + 2*x) / (6 + 3*x) = (3 + 2*x) / (5 + 3*x) → x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_x_value_l56_5656


namespace NUMINAMATH_CALUDE_student_number_problem_l56_5678

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l56_5678


namespace NUMINAMATH_CALUDE_not_parabola_l56_5618

-- Define the equation
def equation (α : Real) (x y : Real) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Theorem statement
theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    equation α x y ↔ y = a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l56_5618


namespace NUMINAMATH_CALUDE_people_disliking_tv_and_books_l56_5660

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def book_and_tv_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_tv_and_books :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * book_and_tv_dislike_percentage⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_people_disliking_tv_and_books_l56_5660


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l56_5687

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define what it means for a function to be increasing on an interval
def IsIncreasingOn (g : ℝ → ℝ) (lb : ℝ) : Prop :=
  ∀ x y, lb ≤ x → x < y → g x < g y

theorem sufficient_not_necessary_condition :
  (IsIncreasingOn (f 1) 2) ∧
  (∃ a : ℝ, a ≠ 1 ∧ IsIncreasingOn (f a) 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l56_5687


namespace NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l56_5626

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- Calculate the lateral surface area of a regular square pyramid -/
def lateral_surface_area (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a regular square pyramid with 
    base edge length 2√2 cm and volume 8 cm³ is 4√22 cm² -/
theorem pyramid_lateral_surface_area :
  let p : RegularSquarePyramid := ⟨2 * Real.sqrt 2, 8⟩
  lateral_surface_area p = 4 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_lateral_surface_area_l56_5626


namespace NUMINAMATH_CALUDE_cow_manure_plant_height_l56_5616

/-- The height of the cow manure plant given the heights of control and bone meal plants -/
theorem cow_manure_plant_height
  (control_height : ℝ)
  (bone_meal_percentage : ℝ)
  (cow_manure_percentage : ℝ)
  (h1 : control_height = 36)
  (h2 : bone_meal_percentage = 1.25)
  (h3 : cow_manure_percentage = 2) :
  control_height * bone_meal_percentage * cow_manure_percentage = 90 := by
  sorry

#check cow_manure_plant_height

end NUMINAMATH_CALUDE_cow_manure_plant_height_l56_5616


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l56_5668

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : (4 * a.val * b.val - 1) ∣ (4 * a.val^2 - 1)^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l56_5668


namespace NUMINAMATH_CALUDE_train_B_speed_train_B_speed_is_36_l56_5649

-- Define the problem parameters
def train_A_length : ℝ := 125  -- meters
def train_B_length : ℝ := 150  -- meters
def train_A_speed : ℝ := 54    -- km/hr
def crossing_time : ℝ := 11    -- seconds

-- Define the theorem
theorem train_B_speed : ℝ :=
  let total_distance := train_A_length + train_B_length
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  relative_speed_kmph - train_A_speed

-- Prove the theorem
theorem train_B_speed_is_36 : train_B_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_B_speed_train_B_speed_is_36_l56_5649


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l56_5641

/-- If 16x^2 + 40x + b is the square of a binomial, then b = 25 -/
theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (p * x + q)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l56_5641


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_two_l56_5631

theorem solution_set_abs_x_times_x_minus_two (x : ℝ) :
  {x : ℝ | |x| * (x - 2) ≥ 0} = {x : ℝ | x ≥ 2 ∨ x = 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_two_l56_5631


namespace NUMINAMATH_CALUDE_problem_statement_l56_5666

def a₁ (n : ℕ+) : ℤ := n.val^2 - 10*n.val + 23
def a₂ (n : ℕ+) : ℤ := n.val^2 - 9*n.val + 31
def a₃ (n : ℕ+) : ℤ := n.val^2 - 12*n.val + 46

theorem problem_statement :
  (∀ n : ℕ+, Even (a₁ n + a₂ n + a₃ n)) ∧
  (∀ n : ℕ+, (Prime (a₁ n) ∧ Prime (a₂ n) ∧ Prime (a₃ n)) ↔ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l56_5666


namespace NUMINAMATH_CALUDE_simplify_expression_l56_5692

theorem simplify_expression (a b : ℝ) : a + (3*a - 3*b) - (a - 2*b) = 3*a - b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l56_5692


namespace NUMINAMATH_CALUDE_scaled_triangle_area_is_32_l56_5633

/-- The area of a triangle with vertices at (0,0), (-3, 7), and (-7, 3), scaled by a factor of 2 -/
def scaledTriangleArea : ℝ := 32

/-- The scaling factor -/
def scalingFactor : ℝ := 2

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℝ × ℝ) := [(0, 0), (-3, 7), (-7, 3)]

/-- Theorem: The area of the scaled triangle is 32 square units -/
theorem scaled_triangle_area_is_32 :
  scaledTriangleArea = 32 :=
by sorry

end NUMINAMATH_CALUDE_scaled_triangle_area_is_32_l56_5633


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l56_5601

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the carpets placed in it -/
structure CarpetLayout where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given layout -/
def tripleOverlapArea (layout : CarpetLayout) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_overlap_area_is_six (layout : CarpetLayout) 
  (h1 : layout.hallWidth = 10 ∧ layout.hallHeight = 10)
  (h2 : layout.carpet1.width = 6 ∧ layout.carpet1.height = 8)
  (h3 : layout.carpet2.width = 6 ∧ layout.carpet2.height = 6)
  (h4 : layout.carpet3.width = 5 ∧ layout.carpet3.height = 7) :
  tripleOverlapArea layout = 6 :=
sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l56_5601


namespace NUMINAMATH_CALUDE_proportion_problem_l56_5676

/-- Given four real numbers in proportion, where three are known, prove the value of the fourth. -/
theorem proportion_problem (a b c d : ℝ) : 
  (a / b = c / d) →  -- a, b, c, d are in proportion
  (a = 2) →          -- a = 2
  (b = 3) →          -- b = 3
  (d = 6) →          -- d = 6
  (c = 4) :=         -- prove c = 4
by sorry

end NUMINAMATH_CALUDE_proportion_problem_l56_5676


namespace NUMINAMATH_CALUDE_legs_walking_theorem_l56_5652

/-- The number of legs walking on the ground given the conditions of the problem -/
def legs_walking_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_riding := num_horses / 2
  let num_walking_men := num_men - num_riding
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  let walking_horse_legs := horse_legs / 2
  men_legs + walking_horse_legs

/-- Theorem stating that given 14 horses, the number of legs walking on the ground is 42 -/
theorem legs_walking_theorem : legs_walking_on_ground 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_legs_walking_theorem_l56_5652


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l56_5640

theorem complex_exponential_sum (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/4 : ℂ) + (3/7 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/4 : ℂ) - (3/7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l56_5640


namespace NUMINAMATH_CALUDE_scale_length_theorem_l56_5617

/-- Given a scale divided into equal parts, this function calculates its total length -/
def scaleLength (numParts : ℕ) (partLength : ℕ) : ℕ :=
  numParts * partLength

/-- Theorem stating that a scale with 4 parts of 20 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  scaleLength 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_theorem_l56_5617


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l56_5677

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 2 ↔ 3 * x + 4 > 5 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l56_5677


namespace NUMINAMATH_CALUDE_asymptote_parabola_intersection_distance_l56_5688

/-- The distance between the two points where the asymptotes of the hyperbola x^2 - y^2 = 1
    intersect with the parabola y^2 = 4x is 8, given that one intersection point is the origin. -/
theorem asymptote_parabola_intersection_distance : 
  let hyperbola := fun (x y : ℝ) => x^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4*x
  let asymptote1 := fun (x : ℝ) => x
  let asymptote2 := fun (x : ℝ) => -x
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (4, 4)
  let B : ℝ × ℝ := (4, -4)
  (hyperbola O.1 O.2) ∧ 
  (parabola O.1 O.2) ∧
  (parabola A.1 A.2) ∧ 
  (parabola B.1 B.2) ∧
  (A.2 = asymptote1 A.1) ∧
  (B.2 = asymptote2 B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_asymptote_parabola_intersection_distance_l56_5688


namespace NUMINAMATH_CALUDE_a_less_than_b_l56_5637

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) 
  (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l56_5637


namespace NUMINAMATH_CALUDE_cost_difference_l56_5690

def ice_cream_cartons : ℕ := 100
def yoghurt_cartons : ℕ := 35
def ice_cream_cost_per_carton : ℚ := 12
def yoghurt_cost_per_carton : ℚ := 3
def ice_cream_discount_rate : ℚ := 0.05
def yoghurt_tax_rate : ℚ := 0.08

def ice_cream_total_cost : ℚ := ice_cream_cartons * ice_cream_cost_per_carton
def yoghurt_total_cost : ℚ := yoghurt_cartons * yoghurt_cost_per_carton

def ice_cream_discounted_cost : ℚ := ice_cream_total_cost * (1 - ice_cream_discount_rate)
def yoghurt_taxed_cost : ℚ := yoghurt_total_cost * (1 + yoghurt_tax_rate)

theorem cost_difference : 
  ice_cream_discounted_cost - yoghurt_taxed_cost = 1026.60 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l56_5690


namespace NUMINAMATH_CALUDE_mentorship_arrangements_count_l56_5664

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of permutations of k items from n items --/
def permutations (n k : ℕ) : ℕ := sorry

/-- Calculates the number of mentorship arrangements for 5 students and 3 teachers --/
def mentorshipArrangements : ℕ :=
  let studentGroups := choose 5 2 * choose 3 2 * choose 1 1 / 2
  studentGroups * permutations 3 3

theorem mentorship_arrangements_count :
  mentorshipArrangements = 90 := by sorry

end NUMINAMATH_CALUDE_mentorship_arrangements_count_l56_5664


namespace NUMINAMATH_CALUDE_proposition_all_lines_proposition_line_planes_proposition_all_planes_l56_5674

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricObject) : Prop :=
  perpendicular x y ∧ parallel y z → perpendicular x z

-- Theorem for case 1: all lines
theorem proposition_all_lines :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Line ∧ 
  z = GeometricObject.Line →
  proposition x y z :=
sorry

-- Theorem for case 2: x is line, y and z are planes
theorem proposition_line_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

-- Theorem for case 3: all planes
theorem proposition_all_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Plane ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

end NUMINAMATH_CALUDE_proposition_all_lines_proposition_line_planes_proposition_all_planes_l56_5674


namespace NUMINAMATH_CALUDE_total_book_price_l56_5694

/-- Given the following conditions:
  - Total number of books: 90
  - Math books cost: $4 each
  - History books cost: $5 each
  - Number of math books: 60
  Prove that the total price of all books is $390 -/
theorem total_book_price (total_books : Nat) (math_book_price history_book_price : Nat) (math_books : Nat) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books = 60 →
  math_books * math_book_price + (total_books - math_books) * history_book_price = 390 := by
  sorry

#check total_book_price

end NUMINAMATH_CALUDE_total_book_price_l56_5694


namespace NUMINAMATH_CALUDE_homecoming_ticket_sales_l56_5607

theorem homecoming_ticket_sales
  (single_price : ℕ)
  (couple_price : ℕ)
  (total_attendance : ℕ)
  (couple_tickets_sold : ℕ)
  (h1 : single_price = 20)
  (h2 : couple_price = 35)
  (h3 : total_attendance = 128)
  (h4 : couple_tickets_sold = 16) :
  single_price * (total_attendance - 2 * couple_tickets_sold) +
  couple_price * couple_tickets_sold = 2480 := by
sorry


end NUMINAMATH_CALUDE_homecoming_ticket_sales_l56_5607


namespace NUMINAMATH_CALUDE_vacation_cost_l56_5659

theorem vacation_cost (num_people : ℕ) (plane_ticket_cost : ℕ) (hotel_cost_per_day : ℕ) (num_days : ℕ) : 
  num_people = 2 → 
  plane_ticket_cost = 24 → 
  hotel_cost_per_day = 12 → 
  num_days = 3 → 
  num_people * plane_ticket_cost + num_people * hotel_cost_per_day * num_days = 120 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l56_5659


namespace NUMINAMATH_CALUDE_eight_weavers_eight_days_l56_5696

/-- Represents the number of mats woven by a given number of mat-weavers in a given number of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℕ := sorry

/-- The rate at which mat-weavers work is constant. -/
axiom constant_rate : mats_woven 4 4 = 4

/-- Theorem stating that 8 mat-weavers can weave 16 mats in 8 days. -/
theorem eight_weavers_eight_days : mats_woven 8 8 = 16 := by sorry

end NUMINAMATH_CALUDE_eight_weavers_eight_days_l56_5696


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l56_5609

theorem average_of_six_numbers (a b c d e f : ℝ) :
  (a + b + c + d + e + f) / 6 = 4.60 →
  (c + d) / 2 = 3.8 →
  (e + f) / 2 = 6.6 →
  (a + b) / 2 = 3.4 :=
by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l56_5609


namespace NUMINAMATH_CALUDE_pool_draining_rate_l56_5651

/-- Given a rectangular pool with specified dimensions, capacity, and draining time,
    calculate the rate of water removal in cubic feet per minute. -/
theorem pool_draining_rate
  (width : ℝ) (length : ℝ) (depth : ℝ) (capacity : ℝ) (drain_time : ℝ)
  (h_width : width = 50)
  (h_length : length = 150)
  (h_depth : depth = 10)
  (h_capacity : capacity = 0.8)
  (h_drain_time : drain_time = 1000)
  : (width * length * depth * capacity) / drain_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_draining_rate_l56_5651


namespace NUMINAMATH_CALUDE_function_composition_implies_sum_zero_l56_5667

theorem function_composition_implies_sum_zero 
  (a b c d : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : d ≠ 0) 
  (f : ℝ → ℝ) 
  (h5 : ∀ x, f x = (2*a*x + b) / (c*x + 2*d)) 
  (h6 : ∀ x, f (f x) = 3*x - 4) : 
  a + d = 0 := by
sorry

end NUMINAMATH_CALUDE_function_composition_implies_sum_zero_l56_5667


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l56_5613

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℕ)
  (water : ℕ)
  (sugar : ℕ)

/-- Calculates the new ratio based on the original ratio -/
def new_ratio (original : RecipeRatio) : RecipeRatio :=
  { flour := original.flour,
    water := original.flour / 2,
    sugar := original.sugar * 2 }

/-- Calculates the amount of an ingredient based on the ratio and a known amount -/
def calculate_amount (ratio : RecipeRatio) (known_part : ℕ) (known_amount : ℚ) (target_part : ℕ) : ℚ :=
  (known_amount * target_part) / known_part

theorem sugar_amount_in_new_recipe : 
  let original_ratio := RecipeRatio.mk 8 4 3
  let new_ratio := new_ratio original_ratio
  let water_amount : ℚ := 2
  calculate_amount new_ratio new_ratio.water water_amount new_ratio.sugar = 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l56_5613


namespace NUMINAMATH_CALUDE_inequality_proof_l56_5669

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  Real.sqrt (x^2 - x + 1/2) ≥ 1 / (x + 1/x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l56_5669


namespace NUMINAMATH_CALUDE_last_digit_3_count_l56_5670

/-- The last digit of 7^n -/
def last_digit (n : ℕ) : ℕ := (7^n) % 10

/-- Whether the last digit of 7^n is 3 -/
def is_last_digit_3 (n : ℕ) : Prop := last_digit n = 3

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^n whose last digit is 3 -/
def count_last_digit_3 (n : ℕ) : ℕ := (n + 3) / 4

theorem last_digit_3_count :
  count_last_digit_3 2009 = 502 :=
sorry

end NUMINAMATH_CALUDE_last_digit_3_count_l56_5670


namespace NUMINAMATH_CALUDE_path_result_l56_5630

def move_north (x : ℚ) : ℚ := x + 7
def move_east (x : ℚ) : ℚ := x - 4
def move_south (x : ℚ) : ℚ := x / 2
def move_west (x : ℚ) : ℚ := x * 3

def path (x : ℚ) : ℚ :=
  move_north (move_east (move_south (move_west (move_west (move_south (move_east (move_north x)))))))

theorem path_result : path 21 = 57 := by
  sorry

end NUMINAMATH_CALUDE_path_result_l56_5630


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l56_5658

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 4 * a + k = 0) ∧ 
    (5 * b^2 + 4 * b + k = 0) ∧ 
    (|a - b| = a^2 + b^2)) ↔ 
  (k = 3/5 ∨ k = -12/5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l56_5658


namespace NUMINAMATH_CALUDE_intersection_of_sets_l56_5619

theorem intersection_of_sets : 
  let M : Set Int := {-1, 0}
  let N : Set Int := {0, 1}
  M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l56_5619


namespace NUMINAMATH_CALUDE_marble_collection_problem_l56_5604

/-- Represents the number of marbles collected by a person -/
structure MarbleCount where
  red : ℕ
  blue : ℕ

/-- The marble collection problem -/
theorem marble_collection_problem 
  (mary jenny anie tom : MarbleCount)
  (h1 : mary.red = 2 * jenny.red)
  (h2 : mary.blue = anie.blue / 2)
  (h3 : anie.red = mary.red + 20)
  (h4 : anie.blue = 2 * jenny.blue)
  (h5 : tom.red = anie.red + 10)
  (h6 : tom.blue = mary.blue)
  (h7 : jenny.red = 30)
  (h8 : jenny.blue = 25) :
  mary.blue + jenny.blue + anie.blue + tom.blue = 125 := by
  sorry

end NUMINAMATH_CALUDE_marble_collection_problem_l56_5604


namespace NUMINAMATH_CALUDE_two_numbers_difference_l56_5695

theorem two_numbers_difference (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x < y ∧ 
  (Finset.sum (Finset.range 38) id) - x - y = x * y →
  y - x = 10 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l56_5695


namespace NUMINAMATH_CALUDE_two_cars_total_distance_l56_5675

/-- Proves that given two cars with specified fuel efficiencies and consumption,
    the total distance driven is 1750 miles. -/
theorem two_cars_total_distance
  (efficiency1 : ℝ) (efficiency2 : ℝ) (total_consumption : ℝ) (consumption1 : ℝ)
  (h1 : efficiency1 = 25)
  (h2 : efficiency2 = 40)
  (h3 : total_consumption = 55)
  (h4 : consumption1 = 30) :
  efficiency1 * consumption1 + efficiency2 * (total_consumption - consumption1) = 1750 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_total_distance_l56_5675


namespace NUMINAMATH_CALUDE_exactly_one_solves_l56_5698

/-- The probability that exactly one person solves a problem given two independent probabilities -/
theorem exactly_one_solves (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = p₁ + p₂ - 2 * p₁ * p₂ := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_solves_l56_5698


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l56_5648

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l56_5648


namespace NUMINAMATH_CALUDE_expected_score_is_correct_l56_5646

/-- The expected score for a round in the basketball shooting game. -/
def expected_score : ℝ := 6

/-- The probability of making a shot. -/
def shot_probability : ℝ := 0.5

/-- The score for making the first shot. -/
def first_shot_score : ℕ := 8

/-- The score for making the second shot (after missing the first). -/
def second_shot_score : ℕ := 6

/-- The score for making the third shot (after missing the first two). -/
def third_shot_score : ℕ := 4

/-- The score for missing all three shots. -/
def miss_all_score : ℕ := 0

/-- Theorem stating that the expected score is correct given the game rules. -/
theorem expected_score_is_correct :
  expected_score = 
    shot_probability * first_shot_score +
    (1 - shot_probability) * shot_probability * second_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * shot_probability * third_shot_score +
    (1 - shot_probability) * (1 - shot_probability) * (1 - shot_probability) * miss_all_score :=
by sorry

end NUMINAMATH_CALUDE_expected_score_is_correct_l56_5646


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l56_5634

theorem condition_necessary_not_sufficient (b : ℝ) (hb : b ≠ 0) :
  (∃ a : ℝ, a > b ∧ Real.log (a - b) ≤ 0) ∧
  (∀ a : ℝ, Real.log (a - b) > 0 → a > b) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l56_5634


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l56_5662

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people who must sit together -/
def must_sit_together : ℕ := 2

/-- The number of people available to sit next to the fixed person -/
def available_neighbors : ℕ := total_people - must_sit_together - 1

/-- The number of neighbors to choose -/
def neighbors_to_choose : ℕ := 2

theorem circular_seating_arrangement :
  Nat.choose available_neighbors neighbors_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l56_5662


namespace NUMINAMATH_CALUDE_bobby_candy_chocolate_difference_l56_5605

/-- Given the number of candy pieces Bobby ate initially and additionally,
    as well as the number of chocolate pieces, prove that Bobby ate 58 more
    pieces of candy than chocolate. -/
theorem bobby_candy_chocolate_difference
  (initial_candy : ℕ)
  (additional_candy : ℕ)
  (chocolate : ℕ)
  (h1 : initial_candy = 38)
  (h2 : additional_candy = 36)
  (h3 : chocolate = 16) :
  initial_candy + additional_candy - chocolate = 58 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_chocolate_difference_l56_5605


namespace NUMINAMATH_CALUDE_ordering_of_exponentials_l56_5603

theorem ordering_of_exponentials :
  let a : ℝ := 2^(2/3)
  let b : ℝ := 2^(2/5)
  let c : ℝ := 3^(2/3)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponentials_l56_5603


namespace NUMINAMATH_CALUDE_cupcake_price_is_one_fifty_l56_5638

/-- Represents the daily production and prices of bakery items -/
structure BakeryProduction where
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ
  cookie_price_per_packet : ℚ
  biscuit_price_per_packet : ℚ

/-- Calculates the price of a cupcake given the bakery production and total earnings -/
def calculate_cupcake_price (prod : BakeryProduction) (days : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_cookies_earnings := prod.cookie_packets_per_day * days * prod.cookie_price_per_packet
  let total_biscuits_earnings := prod.biscuit_packets_per_day * days * prod.biscuit_price_per_packet
  let cupcakes_earnings := total_earnings - total_cookies_earnings - total_biscuits_earnings
  cupcakes_earnings / (prod.cupcakes_per_day * days)

/-- Theorem stating that the cupcake price is $1.50 given the specified conditions -/
theorem cupcake_price_is_one_fifty :
  let prod : BakeryProduction := {
    cupcakes_per_day := 20,
    cookie_packets_per_day := 10,
    biscuit_packets_per_day := 20,
    cookie_price_per_packet := 2,
    biscuit_price_per_packet := 1
  }
  let days : ℕ := 5
  let total_earnings : ℚ := 350
  calculate_cupcake_price prod days total_earnings = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_cupcake_price_is_one_fifty_l56_5638


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l56_5661

theorem quadratic_rewrite_ratio (b c : ℝ) :
  (∀ x, x^2 + 1300*x + 1300 = (x + b)^2 + c) →
  c / b = -648 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l56_5661


namespace NUMINAMATH_CALUDE_cooking_shopping_combinations_l56_5689

theorem cooking_shopping_combinations (n : ℕ) (k : ℕ) (h : n = 5 ∧ k = 3) : 
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_shopping_combinations_l56_5689


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l56_5628

def decimal_digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_theorem : 
  decimal_digit_sum (2^2007 * 5^2005 * 7) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l56_5628


namespace NUMINAMATH_CALUDE_small_boxes_count_l56_5672

theorem small_boxes_count (total_bars : ℕ) (bars_per_box : ℕ) (h1 : total_bars = 640) (h2 : bars_per_box = 32) :
  total_bars / bars_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l56_5672


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l56_5639

/-- The equation of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- The equation of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := y = -x - 4

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

/-- Theorem stating that the given symmetric circle is correct -/
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ symmetry_line ((x + x₀)/2) ((y + y₀)/2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l56_5639


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l56_5625

theorem arcsin_one_half_equals_pi_over_six : 
  Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l56_5625


namespace NUMINAMATH_CALUDE_mark_marbles_count_l56_5621

def connie_marbles : ℕ := 323
def juan_marbles : ℕ := connie_marbles + 175
def mark_marbles : ℕ := juan_marbles * 3

theorem mark_marbles_count : mark_marbles = 1494 := by
  sorry

end NUMINAMATH_CALUDE_mark_marbles_count_l56_5621


namespace NUMINAMATH_CALUDE_initial_average_marks_l56_5643

theorem initial_average_marks (n : ℕ) (wrong_mark correct_mark : ℝ) (correct_average : ℝ) :
  n = 10 ∧ wrong_mark = 60 ∧ correct_mark = 10 ∧ correct_average = 95 →
  (n * correct_average - (wrong_mark - correct_mark)) / n = 90 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l56_5643


namespace NUMINAMATH_CALUDE_train_crossing_time_train_crossing_platform_time_l56_5602

/-- Calculates the time required for a train to cross a platform --/
theorem train_crossing_time 
  (train_speed_kmph : ℝ) 
  (man_crossing_time : ℝ) 
  (platform_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

/-- Proves that the train takes 30 seconds to cross the platform --/
theorem train_crossing_platform_time : 
  train_crossing_time 72 19 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_train_crossing_platform_time_l56_5602


namespace NUMINAMATH_CALUDE_sara_marble_count_l56_5691

/-- The number of black marbles Sara has after receiving a gift from Fred -/
def saras_marbles (initial : Float) (gift : Float) : Float :=
  initial + gift

/-- Theorem stating that Sara has 1025.0 black marbles after receiving Fred's gift -/
theorem sara_marble_count : saras_marbles 792.0 233.0 = 1025.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_marble_count_l56_5691


namespace NUMINAMATH_CALUDE_clothing_prices_correct_l56_5671

/-- Represents the price of a clothing item -/
structure ClothingPrice where
  purchase : ℝ
  marked : ℝ

/-- Solves the clothing price problem given the conditions -/
def solve_clothing_prices (total_paid markup_percent discount_a discount_b total_marked : ℝ) : 
  (ClothingPrice × ClothingPrice) :=
  let a : ClothingPrice := { 
    purchase := 50,
    marked := 70
  }
  let b : ClothingPrice := {
    purchase := 100,
    marked := 140
  }
  (a, b)

/-- Theorem stating the correctness of the solution -/
theorem clothing_prices_correct 
  (total_paid : ℝ)
  (markup_percent : ℝ)
  (discount_a : ℝ)
  (discount_b : ℝ)
  (total_marked : ℝ)
  (h1 : total_paid = 182)
  (h2 : markup_percent = 40)
  (h3 : discount_a = 80)
  (h4 : discount_b = 90)
  (h5 : total_marked = 210) :
  let (a, b) := solve_clothing_prices total_paid markup_percent discount_a discount_b total_marked
  (a.purchase = 50 ∧ 
   a.marked = 70 ∧ 
   b.purchase = 100 ∧ 
   b.marked = 140 ∧
   a.marked + b.marked = total_marked ∧
   (discount_a / 100) * a.marked + (discount_b / 100) * b.marked = total_paid ∧
   a.marked = a.purchase * (1 + markup_percent / 100) ∧
   b.marked = b.purchase * (1 + markup_percent / 100)) := by
  sorry


end NUMINAMATH_CALUDE_clothing_prices_correct_l56_5671


namespace NUMINAMATH_CALUDE_horner_rule_operations_l56_5655

/-- Horner's Rule representation of a polynomial -/
def horner_representation (coeffs : List ℤ) : ℤ → ℤ :=
  fun x => coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's Rule evaluation -/
def horner_operation_count (coeffs : List ℤ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 4x^5 - 3x^4 + 6x - 9 -/
def f : List ℤ := [4, -3, 0, 0, 6, -9]

theorem horner_rule_operations :
  horner_operation_count f = (5, 3) := by sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l56_5655


namespace NUMINAMATH_CALUDE_impossibleConfiguration_l56_5622

/-- Represents a configuration of points on a circle -/
structure CircleConfiguration where
  numPoints : ℕ
  circumference : ℕ

/-- Checks if a configuration satisfies the arc length condition -/
def satisfiesArcLengthCondition (config : CircleConfiguration) : Prop :=
  ∃ (points : Fin config.numPoints → ℝ),
    (∀ i, 0 ≤ points i ∧ points i < config.circumference) ∧
    (∀ l : ℕ, 1 ≤ l ∧ l < config.circumference →
      ∃ i j, (points j - points i + config.circumference) % config.circumference = l)

/-- The main theorem stating the impossibility of the configuration -/
theorem impossibleConfiguration :
  ¬ satisfiesArcLengthCondition ⟨10, 90⟩ := by
  sorry


end NUMINAMATH_CALUDE_impossibleConfiguration_l56_5622


namespace NUMINAMATH_CALUDE_triangle_BC_length_l56_5657

/-- Triangle ABC with given properties -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  AM : ℝ  -- Median from A to midpoint of BC
  area : ℝ
  BC : ℝ

/-- The triangle satisfies the given conditions -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.AB = 6 ∧ t.AC = 8 ∧ t.AM = 5 ∧ t.area = 24

/-- Theorem: If a triangle satisfies the given conditions, its BC side length is 10 -/
theorem triangle_BC_length (t : Triangle) (h : satisfies_conditions t) : t.BC = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_length_l56_5657


namespace NUMINAMATH_CALUDE_equation_2010_l56_5684

theorem equation_2010 (digits : Finset Nat) : digits = {2, 3, 5, 6, 7} →
  ∃ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  ((d = 67 ∧ (6 ∈ digits ∧ 7 ∈ digits)) ∨ d ∈ digits) ∧
  a * b * c * d = 2010 :=
by
  sorry

#check equation_2010

end NUMINAMATH_CALUDE_equation_2010_l56_5684


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l56_5682

theorem consecutive_integers_product_812_sum_57 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 812 → n + (n + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l56_5682


namespace NUMINAMATH_CALUDE_factor_implies_values_l56_5685

theorem factor_implies_values (p q : ℝ) : 
  (∃ (a b c : ℝ), (X^4 + p*X^2 + q) = (X^2 + 2*X + 5) * (a*X^2 + b*X + c)) → 
  p = 6 ∧ q = 25 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_values_l56_5685


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l56_5632

/-- Represents a right triangular pyramid -/
structure RightTriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The area of the base of the pyramid -/
  baseArea : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right triangular pyramid, if two cross sections parallel to the base 
have areas of 144√3 and 324√3 square cm, and are 6 cm apart, 
then the larger cross section is 18 cm from the apex.
-/
theorem larger_cross_section_distance (pyramid : RightTriangularPyramid) 
  (section1 section2 : CrossSection) : 
  section1.area = 144 * Real.sqrt 3 →
  section2.area = 324 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 18 := by
  sorry

#check larger_cross_section_distance

end NUMINAMATH_CALUDE_larger_cross_section_distance_l56_5632


namespace NUMINAMATH_CALUDE_parallel_planes_from_intersecting_parallel_lines_l56_5606

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the property of two lines intersecting
variable (lines_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_intersecting_parallel_lines
  (α β : Plane) (l₁ l₂ : Line)
  (h1 : line_in_plane l₁ α)
  (h2 : line_in_plane l₂ α)
  (h3 : lines_intersect l₁ l₂)
  (h4 : line_parallel_to_plane l₁ β)
  (h5 : line_parallel_to_plane l₂ β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_intersecting_parallel_lines_l56_5606


namespace NUMINAMATH_CALUDE_pigeon_problem_l56_5644

/-- The number of pigeons in a group with the following properties:
  1. When each pigeonhole houses 6 pigeons, 3 pigeons are left without a pigeonhole.
  2. When 5 more pigeons arrive, each pigeonhole fits exactly 8 pigeons. -/
def original_pigeons : ℕ := 27

/-- The number of pigeonholes available. -/
def pigeonholes : ℕ := 3

theorem pigeon_problem :
  (6 * pigeonholes + 3 = original_pigeons) ∧
  (8 * pigeonholes = original_pigeons + 5) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_problem_l56_5644


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l56_5683

def moonglow_pumpkins : ℕ := 14

def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l56_5683


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l56_5624

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 65 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l56_5624


namespace NUMINAMATH_CALUDE_cosine_arithmetic_geometric_progression_l56_5686

theorem cosine_arithmetic_geometric_progression (x y z : ℝ) (α : ℝ) :
  (∃ d : ℝ, d ≠ 0 ∧ (1 + Real.cos x) * d = 1 + Real.cos y ∧ (1 + Real.cos y) * d = 1 + Real.cos z) →
  x = y - α ∧ z = y + α →
  α = Real.arccos (5/9) →
  Real.cos y = -7/9 := by
sorry

end NUMINAMATH_CALUDE_cosine_arithmetic_geometric_progression_l56_5686


namespace NUMINAMATH_CALUDE_guanaco_numbers_l56_5681

def is_guanaco (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = a * 1000 + b * 100 + c * 10 + d ∧
    (a * 10 + b) * (c * 10 + d) ∣ n

theorem guanaco_numbers :
  ∀ n : ℕ, is_guanaco n ↔ (n = 1352 ∨ n = 1734) :=
by sorry

end NUMINAMATH_CALUDE_guanaco_numbers_l56_5681


namespace NUMINAMATH_CALUDE_trapezoid_area_l56_5650

/-- Given four identical trapezoids that form a square, prove the area of each trapezoid --/
theorem trapezoid_area (base_small : ℝ) (base_large : ℝ) (square_area : ℝ) :
  base_small = 30 →
  base_large = 50 →
  square_area = 2500 →
  (∃ (trapezoid_area : ℝ), 
    trapezoid_area = (square_area - base_small ^ 2) / 4 ∧ 
    trapezoid_area = 400) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l56_5650


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l56_5614

theorem no_prime_satisfies_equation : ¬∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (2 * p^2 + 5 * p + 3) + (5 * p^2 + p + 2) + (p^2 + 1) + (2 * p^2 + 4 * p + 3) + (p^2 + 6) = 
  (7 * p^2 + 6 * p + 5) + (4 * p^2 + 3 * p + 2) + (p^2 + 2 * p) := by
  sorry

#check no_prime_satisfies_equation

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l56_5614


namespace NUMINAMATH_CALUDE_common_external_tangents_parallel_l56_5653

/-- Two circles with equal radii -/
structure EqualRadiiCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius : ℝ

/-- A line representing a common external tangent -/
structure CommonExternalTangent where
  slope : ℝ
  intercept : ℝ

/-- The line connecting the centers of two circles -/
def centerLine (c : EqualRadiiCircles) : ℝ × ℝ → Prop :=
  fun p => ∃ t : ℝ, p = (1 - t) • c.center1 + t • c.center2

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  ∀ p q : ℝ × ℝ, l1 p → l1 q → l2 p → l2 q → 
    (p.1 - q.1) * (p.2 - q.2) = (p.1 - q.1) * (p.2 - q.2)

theorem common_external_tangents_parallel (c : EqualRadiiCircles) 
  (t1 t2 : CommonExternalTangent) : 
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (fun p => p.2 = t2.slope * p.1 + t2.intercept) ∧
  parallel (fun p => p.2 = t1.slope * p.1 + t1.intercept) 
           (centerLine c) := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangents_parallel_l56_5653


namespace NUMINAMATH_CALUDE_prism_surface_area_l56_5629

theorem prism_surface_area (R : ℝ) (h : ℝ) :
  R > 0 →
  (R / 2)^2 + 3 = R^2 →
  2 + h^2 = 4 * R^2 →
  2 + 4 * h = 4 * Real.sqrt 14 + 2 :=
by sorry

end NUMINAMATH_CALUDE_prism_surface_area_l56_5629


namespace NUMINAMATH_CALUDE_cone_volume_maximization_l56_5635

theorem cone_volume_maximization (x : Real) : 
  let r := 1 -- radius of the original circular plate
  let cone_base_radius := (2 * Real.pi - x) / (2 * Real.pi) * r
  let cone_height := Real.sqrt (r ^ 2 - cone_base_radius ^ 2)
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius ^ 2 * cone_height
  (∀ y, cone_volume ≤ (let cone_base_radius := (2 * Real.pi - y) / (2 * Real.pi) * r
                       let cone_height := Real.sqrt (r ^ 2 - cone_base_radius ^ 2)
                       (1 / 3) * Real.pi * cone_base_radius ^ 2 * cone_height)) →
  x = (6 - 2 * Real.sqrt 6) / 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_maximization_l56_5635


namespace NUMINAMATH_CALUDE_prob_no_adjacent_three_of_ten_l56_5699

/-- The number of chairs in a row -/
def n : ℕ := 10

/-- The number of people choosing seats -/
def k : ℕ := 3

/-- The probability of k people choosing seats from n chairs such that none sit next to each other -/
def prob_no_adjacent (n k : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the probability of 3 people choosing seats from 10 chairs 
    such that none sit next to each other is 1/3 -/
theorem prob_no_adjacent_three_of_ten : prob_no_adjacent n k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_three_of_ten_l56_5699


namespace NUMINAMATH_CALUDE_hippocrates_lunes_l56_5697

theorem hippocrates_lunes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + b^2 = c^2) :
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let triangle_area := a * b / 2
  let lunes_area := semicircle_area a + semicircle_area b - (semicircle_area c - triangle_area)
  lunes_area = triangle_area := by
sorry

end NUMINAMATH_CALUDE_hippocrates_lunes_l56_5697


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l56_5600

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 140*x - 4000

/-- Represents the valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 50 ≤ x ∧ x ≤ 100

theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    valid_price_range x ∧ 
    (∀ y, valid_price_range y → profit_function y ≤ profit_function x) ∧
    x = 70 ∧ 
    profit_function x = 900 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l56_5600


namespace NUMINAMATH_CALUDE_impossible_digit_product_after_increment_l56_5654

def digit_product (n : ℕ) : ℕ := sorry

theorem impossible_digit_product_after_increment :
  ∀ N : ℕ,
  N > 0 →
  digit_product N = 20 →
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 24) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 25) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 30) ∧
  (∃ M : ℕ, M = N + 1 ∧ digit_product M = 40) ∧
  ¬(∃ M : ℕ, M = N + 1 ∧ digit_product M = 35) :=
by sorry

end NUMINAMATH_CALUDE_impossible_digit_product_after_increment_l56_5654


namespace NUMINAMATH_CALUDE_largest_value_l56_5665

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > a^2 + b^2 ∧ b > 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l56_5665


namespace NUMINAMATH_CALUDE_unique_integer_solution_l56_5623

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 3178 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l56_5623


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l56_5610

/-- Given two lines that intersect at x = -10, prove the value of k -/
theorem intersection_point_k_value :
  let line1 : ℝ → ℝ → ℝ := λ x y => -3 * x + y
  let line2 : ℝ → ℝ → ℝ := λ x y => 0.75 * x + y
  let k : ℝ := line1 (-10) (line2 (-10) 20)
  k = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l56_5610


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l56_5680

/-- The area of the overlapping region of two sectors in a circle -/
theorem overlapping_sectors_area (r : ℝ) (θ₁ θ₂ : ℝ) (h_r : r = 10) (h_θ₁ : θ₁ = 45) (h_θ₂ : θ₂ = 90) :
  let sector_area (θ : ℝ) := (θ / 360) * π * r^2
  min (sector_area θ₁) (sector_area θ₂) = 12.5 * π :=
by sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l56_5680


namespace NUMINAMATH_CALUDE_brick_width_calculation_l56_5663

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ brick_width : ℝ,
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = brick_length * brick_width * total_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l56_5663


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l56_5673

theorem quadratic_roots_sum_and_product :
  let f : ℝ → ℝ := λ x => x^2 - 18*x + 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ 
    (x₁ + x₂ = 18) ∧ (x₁ * x₂ = 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l56_5673


namespace NUMINAMATH_CALUDE_james_writing_time_l56_5679

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  total_pages : ℕ
  total_weeks : ℕ

/-- Calculate the hours James writes per night -/
def hours_per_night (s : WritingScenario) : ℚ :=
  (s.total_pages : ℚ) / (s.total_weeks * 7 * s.pages_per_hour)

/-- Theorem stating that James writes for 3 hours every night -/
theorem james_writing_time (s : WritingScenario)
  (h1 : s.pages_per_hour = 5)
  (h2 : s.total_pages = 735)
  (h3 : s.total_weeks = 7) :
  hours_per_night s = 3 := by
  sorry

#eval hours_per_night ⟨5, 735, 7⟩

end NUMINAMATH_CALUDE_james_writing_time_l56_5679


namespace NUMINAMATH_CALUDE_choose_two_correct_l56_5636

/-- The number of ways to choose 2 different items from n distinct items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that choose_two gives the correct number of ways to choose 2 from n -/
theorem choose_two_correct (n : ℕ) : choose_two n = Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_correct_l56_5636


namespace NUMINAMATH_CALUDE_pieces_with_high_product_bound_l56_5620

/-- Represents an infinite chessboard with pieces placed on it. -/
structure InfiniteChessboard where
  m : ℕ  -- Total number of pieces
  piece_positions : Finset (ℕ × ℕ)  -- Positions of pieces
  piece_count : piece_positions.card = m  -- Ensure the number of pieces matches m

/-- Calculates the number of pieces in a given row -/
def pieces_in_row (board : InfiniteChessboard) (row : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.1 = row)).card

/-- Calculates the number of pieces in a given column -/
def pieces_in_column (board : InfiniteChessboard) (col : ℕ) : ℕ :=
  (board.piece_positions.filter (fun p => p.2 = col)).card

/-- Calculates the product of pieces in the row and column for a given position -/
def product_for_position (board : InfiniteChessboard) (pos : ℕ × ℕ) : ℕ :=
  (pieces_in_row board pos.1) * (pieces_in_column board pos.2)

/-- The main theorem to be proved -/
theorem pieces_with_high_product_bound (board : InfiniteChessboard) :
  (board.piece_positions.filter (fun pos => product_for_position board pos ≥ 10 * board.m)).card ≤ board.m / 10 :=
sorry

end NUMINAMATH_CALUDE_pieces_with_high_product_bound_l56_5620


namespace NUMINAMATH_CALUDE_fifth_grade_total_is_144_l56_5611

/-- The number of students in the fifth grade of Longteng Primary School --/
def fifth_grade_total : ℕ :=
  let class1 : ℕ := 42
  let class2 : ℕ := (class1 * 6) / 7
  let class3 : ℕ := (class2 * 5) / 6
  let class4 : ℕ := (class3 * 12) / 10
  class1 + class2 + class3 + class4

theorem fifth_grade_total_is_144 : fifth_grade_total = 144 := by
  sorry

end NUMINAMATH_CALUDE_fifth_grade_total_is_144_l56_5611


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_neither_sufficient_nor_necessary_l56_5612

theorem a_squared_gt_b_squared_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_neither_sufficient_nor_necessary_l56_5612


namespace NUMINAMATH_CALUDE_system_solution_l56_5645

theorem system_solution :
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -0.75) ∧ 
    (5 * x + 3 * y = 5.35) ∧ 
    (abs (x - 0.5111) < 0.0001) ∧ 
    (abs (y - 0.9315) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l56_5645


namespace NUMINAMATH_CALUDE_minimum_races_for_top_three_l56_5693

/-- Represents a horse in the race -/
structure Horse : Type :=
  (id : Nat)

/-- Represents a race with at most 5 horses -/
structure Race : Type :=
  (participants : Finset Horse)
  (size_constraint : participants.card ≤ 5)

/-- The set of all horses -/
def all_horses : Finset Horse := sorry

/-- The proposition that a given number of races is sufficient to determine the top 3 fastest horses -/
def can_determine_top_three (n : Nat) : Prop := sorry

/-- The proposition that a given number of races is necessary to determine the top 3 fastest horses -/
def is_necessary (n : Nat) : Prop := sorry

theorem minimum_races_for_top_three :
  (all_horses.card = 25) →
  (can_determine_top_three 7) ∧
  (∀ m : Nat, m < 7 → ¬(can_determine_top_three m)) ∧
  (is_necessary 7) :=
sorry

end NUMINAMATH_CALUDE_minimum_races_for_top_three_l56_5693


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l56_5642

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l56_5642


namespace NUMINAMATH_CALUDE_smallest_constant_for_triangle_sides_l56_5615

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem smallest_constant_for_triangle_sides (t : Triangle) :
  (t.a^2 + t.b^2) / (t.a * t.b) ≥ 2 ∧
  ∀ N, (∀ t' : Triangle, (t'.a^2 + t'.b^2) / (t'.a * t'.b) < N) → N ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_for_triangle_sides_l56_5615


namespace NUMINAMATH_CALUDE_conference_handshakes_l56_5647

theorem conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) :
  total = 50 →
  group_a = 30 →
  group_b = 20 →
  group_a + group_b = total →
  (group_a * group_b) + (group_b * (group_b - 1) / 2) = 790 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l56_5647
