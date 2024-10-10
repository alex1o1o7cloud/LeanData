import Mathlib

namespace area_of_triangle_fyh_l1867_186748

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  ef : ℝ
  gh : ℝ
  area : ℝ

/-- Theorem: Area of triangle FYH in a trapezoid with specific measurements -/
theorem area_of_triangle_fyh (t : Trapezoid) 
  (h1 : t.ef = 24)
  (h2 : t.gh = 36)
  (h3 : t.area = 360) :
  let height : ℝ := 2 * t.area / (t.ef + t.gh)
  let area_egh : ℝ := (1 / 2) * t.gh * height
  let area_efh : ℝ := t.area - area_egh
  let height_eyh : ℝ := (2 / 5) * height
  let area_efh_recalc : ℝ := (1 / 2) * t.ef * (height - height_eyh)
  area_efh - area_efh_recalc = 86.4 := by
  sorry

end area_of_triangle_fyh_l1867_186748


namespace correct_categorization_l1867_186752

def numbers : List ℚ := [2020, 1, -1, -2021, 1/2, 1/10, -1/3, -3/4, 0, 1/5]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_positive_integer (q : ℚ) : Prop := ∃ (n : ℕ), q = n ∧ n > 0
def is_negative_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n ∧ n < 0
def is_positive_fraction (q : ℚ) : Prop := q > 0 ∧ q < 1
def is_negative_fraction (q : ℚ) : Prop := q < 0 ∧ q > -1

def integers : List ℚ := [2020, 1, -1, -2021, 0]
def positive_integers : List ℚ := [2020, 1]
def negative_integers : List ℚ := [-1, -2021]
def positive_fractions : List ℚ := [1/2, 1/10, 1/5]
def negative_fractions : List ℚ := [-1/3, -3/4]

theorem correct_categorization :
  (∀ q ∈ integers, is_integer q) ∧
  (∀ q ∈ positive_integers, is_positive_integer q) ∧
  (∀ q ∈ negative_integers, is_negative_integer q) ∧
  (∀ q ∈ positive_fractions, is_positive_fraction q) ∧
  (∀ q ∈ negative_fractions, is_negative_fraction q) ∧
  (∀ q ∈ numbers, 
    (is_integer q → q ∈ integers) ∧
    (is_positive_integer q → q ∈ positive_integers) ∧
    (is_negative_integer q → q ∈ negative_integers) ∧
    (is_positive_fraction q → q ∈ positive_fractions) ∧
    (is_negative_fraction q → q ∈ negative_fractions)) :=
by sorry

end correct_categorization_l1867_186752


namespace greater_number_proof_l1867_186785

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 36) (h_diff : a - b = 8) (h_greater : a > b) : a = 22 := by
  sorry

end greater_number_proof_l1867_186785


namespace function_is_even_l1867_186775

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_is_even
  (f : ℝ → ℝ)
  (h1 : has_period f 4)
  (h2 : ∀ x, f (2 + x) = f (2 - x)) :
  is_even_function f :=
sorry

end function_is_even_l1867_186775


namespace empty_solution_set_range_l1867_186742

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x| + |x - 1| < a)) → a ∈ Set.Iic 1 := by
  sorry

end empty_solution_set_range_l1867_186742


namespace land_profit_calculation_l1867_186759

/-- Represents the profit calculation for land distribution among sons -/
theorem land_profit_calculation (total_land : ℝ) (num_sons : ℕ) 
  (profit_per_unit : ℝ) (unit_area : ℝ) (hectare_to_sqm : ℝ) : 
  total_land = 3 ∧ 
  num_sons = 8 ∧ 
  profit_per_unit = 500 ∧ 
  unit_area = 750 ∧ 
  hectare_to_sqm = 10000 → 
  (total_land * hectare_to_sqm / num_sons / unit_area * profit_per_unit * 4 : ℝ) = 10000 := by
  sorry

#check land_profit_calculation

end land_profit_calculation_l1867_186759


namespace parabola_max_ratio_l1867_186760

theorem parabola_max_ratio (p : ℝ) (h : p > 0) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 / 4 ∧
  ∀ (x y : ℝ), y^2 = 2*p*x →
    Real.sqrt (x^2 + y^2) / Real.sqrt ((x - p/6)^2 + y^2) ≤ max :=
by sorry

end parabola_max_ratio_l1867_186760


namespace sin_product_equals_one_sixteenth_l1867_186711

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end sin_product_equals_one_sixteenth_l1867_186711


namespace abs_squared_minus_two_abs_minus_fifteen_solution_set_l1867_186708

theorem abs_squared_minus_two_abs_minus_fifteen_solution_set :
  {x : ℝ | |x|^2 - 2*|x| - 15 > 0} = {x : ℝ | x < -5 ∨ x > 5} := by
  sorry

end abs_squared_minus_two_abs_minus_fifteen_solution_set_l1867_186708


namespace min_value_squared_sum_l1867_186715

theorem min_value_squared_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  x^2 + y^2 ≥ 2 := by
  sorry

end min_value_squared_sum_l1867_186715


namespace triangle_side_values_l1867_186740

/-- Given a triangle ABC with area S, sides b and c, prove that the third side a
    has one of two specific values. -/
theorem triangle_side_values (S b c : ℝ) (h1 : S = 12 * Real.sqrt 3)
    (h2 : b * c = 48) (h3 : b - c = 2) :
    ∃ (a : ℝ), (a = 2 * Real.sqrt 13 ∨ a = 2 * Real.sqrt 37) ∧
               (S = 1/2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) := by
  sorry

end triangle_side_values_l1867_186740


namespace mary_lambs_count_l1867_186709

def lambs_problem (initial_lambs : ℕ) (mother_lambs : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (extra_lambs : ℕ) : Prop :=
  let new_babies := mother_lambs * babies_per_lamb
  let after_births := initial_lambs + new_babies
  let after_trade := after_births - traded_lambs
  let final_count := after_trade + extra_lambs
  final_count = 34

theorem mary_lambs_count : 
  lambs_problem 12 4 3 5 15 := by
  sorry

end mary_lambs_count_l1867_186709


namespace greatest_integer_problem_l1867_186710

theorem greatest_integer_problem (n : ℕ) : n < 50 ∧
  (∃ a : ℤ, n = 6 * a - 1) ∧
  (∃ b : ℤ, n = 8 * b - 5) ∧
  (∃ c : ℤ, n = 3 * c + 2) ∧
  (∀ m : ℕ, m < 50 →
    (∃ a : ℤ, m = 6 * a - 1) →
    (∃ b : ℤ, m = 8 * b - 5) →
    (∃ c : ℤ, m = 3 * c + 2) →
    m ≤ n) →
  n = 41 := by
  sorry

end greatest_integer_problem_l1867_186710


namespace simplify_fourth_root_l1867_186765

theorem simplify_fourth_root (x y : ℕ+) :
  (2^6 * 3^5 * 5^2 : ℝ)^(1/4) = x * y^(1/4) →
  x + y = 306 := by
  sorry

end simplify_fourth_root_l1867_186765


namespace equation_solution_l1867_186795

theorem equation_solution : ∃ x : ℚ, (3*x + 5*x = 600 - (4*x + 6*x)) ∧ x = 100/3 := by
  sorry

end equation_solution_l1867_186795


namespace f_inequality_range_l1867_186730

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log 0.3

theorem f_inequality_range (t : ℝ) :
  (∀ x, f x ≤ t^2/4 - t + 1) ↔ t ∈ Set.Iic 1 ∪ Set.Ici 3 :=
sorry

end f_inequality_range_l1867_186730


namespace system_solution_cubic_equation_solution_l1867_186722

-- Problem 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

-- Problem 2: Cubic equation
theorem cubic_equation_solution :
  ∃! x : ℝ, (2 * x - 1)^3 = -8 ∧ x = -1/2 := by
  sorry

end system_solution_cubic_equation_solution_l1867_186722


namespace train_speed_problem_l1867_186773

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 210)
  (h2 : length2 = 260)
  (h3 : speed1 = 40)
  (h4 : time = 16.918646508279338)
  (h5 : length1 > 0)
  (h6 : length2 > 0)
  (h7 : speed1 > 0)
  (h8 : time > 0) :
  ∃ speed2 : ℝ, 
    speed2 > 0 ∧ 
    (length1 + length2) / 1000 = (speed1 + speed2) * (time / 3600) ∧
    speed2 = 60 := by
  sorry

end train_speed_problem_l1867_186773


namespace sphere_surface_area_equals_volume_l1867_186786

/-- For a sphere with radius 3, its surface area is numerically equal to its volume. -/
theorem sphere_surface_area_equals_volume :
  let r : ℝ := 3
  let surface_area : ℝ := 4 * Real.pi * r^2
  let volume : ℝ := (4/3) * Real.pi * r^3
  surface_area = volume := by sorry

end sphere_surface_area_equals_volume_l1867_186786


namespace train_passengers_l1867_186731

/-- The number of people on a train after three stops -/
def people_after_three_stops (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

/-- Theorem stating the number of people on the train after three stops -/
theorem train_passengers : 
  people_after_three_stops 48 12 7 15 9 6 11 = 42 := by
  sorry

end train_passengers_l1867_186731


namespace cats_weight_l1867_186702

/-- The weight of two cats, where one cat weighs 2 kilograms and the other is twice as heavy, is 6 kilograms. -/
theorem cats_weight (weight_cat1 weight_cat2 : ℝ) : 
  weight_cat1 = 2 → weight_cat2 = 2 * weight_cat1 → weight_cat1 + weight_cat2 = 6 := by
  sorry

end cats_weight_l1867_186702


namespace square_side_length_when_area_equals_perimeter_l1867_186720

theorem square_side_length_when_area_equals_perimeter :
  ∃ (a : ℝ), a > 0 ∧ a^2 = 4*a := by
  sorry

end square_side_length_when_area_equals_perimeter_l1867_186720


namespace complex_magnitude_problem_l1867_186737

theorem complex_magnitude_problem (t : ℝ) (h : t > 0) :
  Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 → t = 2 * Real.sqrt 29 := by
  sorry

end complex_magnitude_problem_l1867_186737


namespace four_color_theorem_l1867_186754

/-- Represents a country on the map -/
structure Country where
  borders : ℕ
  border_divisible_by_three : borders % 3 = 0

/-- Represents a map of countries -/
structure Map where
  countries : List Country

/-- Represents a coloring of the map -/
def Coloring := Map → Country → Fin 4

/-- A coloring is proper if no adjacent countries have the same color -/
def is_proper_coloring (m : Map) (c : Coloring) : Prop := sorry

/-- Volynsky's theorem -/
axiom volynsky_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c

/-- Main theorem: If the number of borders of each country on a normal map
    is divisible by 3, then the map can be properly colored with four colors -/
theorem four_color_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c :=
by
  sorry

end four_color_theorem_l1867_186754


namespace local_minimum_implies_a_equals_four_l1867_186777

/-- The function f with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x^2 - a^2 * x

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * x - a^2

theorem local_minimum_implies_a_equals_four :
  ∀ a : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a x ≥ f a 1) →
  f_derivative a 1 = 0 →
  a = 4 := by
  sorry

#check local_minimum_implies_a_equals_four

end local_minimum_implies_a_equals_four_l1867_186777


namespace p_sufficient_not_necessary_q_l1867_186796

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_q_l1867_186796


namespace triangle_inequality_cube_l1867_186744

theorem triangle_inequality_cube (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
sorry

end triangle_inequality_cube_l1867_186744


namespace binomial_multiplication_l1867_186700

theorem binomial_multiplication (x : ℝ) : (4 * x - 3) * (x + 7) = 4 * x^2 + 25 * x - 21 := by
  sorry

end binomial_multiplication_l1867_186700


namespace notebook_pages_calculation_l1867_186770

theorem notebook_pages_calculation (num_notebooks : ℕ) (pages_per_day : ℕ) (days_lasted : ℕ) : 
  num_notebooks > 0 → 
  pages_per_day > 0 → 
  days_lasted > 0 → 
  (pages_per_day * days_lasted) % num_notebooks = 0 → 
  (pages_per_day * days_lasted) / num_notebooks = 40 :=
by
  sorry

#check notebook_pages_calculation 5 4 50

end notebook_pages_calculation_l1867_186770


namespace negation_equivalence_l1867_186793

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) := by
  sorry

end negation_equivalence_l1867_186793


namespace line_plane_relationship_l1867_186792

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection relation between a line and a plane
variable (intersects : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : perpendicular a b)
  (h2 : parallel_line_plane a α) :
  intersects b α ∨ subset_line_plane b α ∨ parallel_line_plane b α :=
sorry

end line_plane_relationship_l1867_186792


namespace unique_solution_for_equation_l1867_186771

theorem unique_solution_for_equation : ∃! (n : ℕ), n > 0 ∧ n^2 + n + 6*n = 210 := by
  sorry

end unique_solution_for_equation_l1867_186771


namespace complex_fraction_power_l1867_186798

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2017 = i := by
  sorry

end complex_fraction_power_l1867_186798


namespace library_book_loan_l1867_186768

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) : 
  initial_books = 75 → 
  return_rate = 4/5 → 
  final_books = 64 → 
  (initial_books : ℚ) - final_books = (1 - return_rate) * 55 := by
  sorry

end library_book_loan_l1867_186768


namespace roses_money_proof_l1867_186790

/-- The amount of money Rose already has -/
def roses_money : ℝ := 7.10

/-- The cost of the paintbrush -/
def paintbrush_cost : ℝ := 2.40

/-- The cost of the set of paints -/
def paints_cost : ℝ := 9.20

/-- The cost of the easel -/
def easel_cost : ℝ := 6.50

/-- The additional amount Rose needs -/
def additional_needed : ℝ := 11

theorem roses_money_proof :
  roses_money + additional_needed = paintbrush_cost + paints_cost + easel_cost :=
by sorry

end roses_money_proof_l1867_186790


namespace geometric_sequence_sum_l1867_186712

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 = 40 →
  a 3 + a 4 = 60 →
  a 5 + a 6 = 90 := by
  sorry

end geometric_sequence_sum_l1867_186712


namespace quadratic_m_range_l1867_186757

def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 - m * x + 1 = 0

def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂

def roots_in_range (m : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_equation m x → x ≥ 1/2 ∧ x ≤ 4

theorem quadratic_m_range :
  ∀ m : ℝ, (has_two_distinct_roots m ∧ roots_in_range m) ↔ (m > 2 * Real.sqrt 2 ∧ m ≤ 3) :=
sorry

end quadratic_m_range_l1867_186757


namespace lg_sum_equals_two_l1867_186738

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : 2 * lg 5 + lg 4 = 2 := by sorry

end lg_sum_equals_two_l1867_186738


namespace geometric_locus_l1867_186746

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y^2 - x^2 = 0
def condition2 (x y : ℝ) : Prop := x^2 + y^2 = 4*(y - 1)
def condition3 (x : ℝ) : Prop := x^2 - 2*x + 1 = 0
def condition4 (x y : ℝ) : Prop := x^2 - 2*x*y + y^2 = -1

-- Define the theorem
theorem geometric_locus :
  (∀ x y : ℝ, condition1 x y ↔ (y = x ∨ y = -x)) ∧
  (∀ x y : ℝ, condition2 x y ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x : ℝ, condition3 x ↔ x = 1) ∧
  (¬∃ x y : ℝ, condition4 x y) :=
by sorry

end geometric_locus_l1867_186746


namespace bike_route_total_length_l1867_186735

/-- The total length of a rectangular bike route -/
def bike_route_length (h1 h2 h3 v1 v2 : ℝ) : ℝ :=
  2 * ((h1 + h2 + h3) + (v1 + v2))

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_total_length :
  bike_route_length 4 7 2 6 7 = 52 := by
  sorry

#eval bike_route_length 4 7 2 6 7

end bike_route_total_length_l1867_186735


namespace ribbon_leftover_l1867_186739

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : ribbon_per_gift = 2) : 
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
  sorry

end ribbon_leftover_l1867_186739


namespace optimal_garden_length_l1867_186706

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  parallel_length : ℝ  -- Length of the side parallel to the house
  perpendicular_length : ℝ  -- Length of the sides perpendicular to the house
  house_length : ℝ  -- Length of the house wall
  fence_cost_per_foot : ℝ  -- Cost of the fence per foot
  total_fence_cost : ℝ  -- Total cost of the fence

/-- The area of the garden. -/
def garden_area (g : Garden) : ℝ :=
  g.parallel_length * g.perpendicular_length

/-- The total length of the fence. -/
def fence_length (g : Garden) : ℝ :=
  g.parallel_length + 2 * g.perpendicular_length

/-- Theorem stating that the optimal garden length is 100 feet. -/
theorem optimal_garden_length (g : Garden) 
    (h1 : g.house_length = 500)
    (h2 : g.fence_cost_per_foot = 10)
    (h3 : g.total_fence_cost = 2000)
    (h4 : fence_length g = g.total_fence_cost / g.fence_cost_per_foot) :
  ∃ (optimal_length : ℝ), 
    optimal_length = 100 ∧ 
    ∀ (other_length : ℝ), 
      0 < other_length → 
      other_length ≤ fence_length g / 2 →
      garden_area { g with parallel_length := other_length, 
                           perpendicular_length := (fence_length g - other_length) / 2 } ≤ 
      garden_area { g with parallel_length := optimal_length, 
                           perpendicular_length := (fence_length g - optimal_length) / 2 } :=
sorry

end optimal_garden_length_l1867_186706


namespace vendor_first_day_sale_percentage_l1867_186767

/-- Represents the percentage of apples sold on the first day -/
def first_day_sale_percentage : ℝ := sorry

/-- Represents the total number of apples initially -/
def total_apples : ℝ := sorry

/-- Represents the number of apples remaining after the first day's sale -/
def apples_after_first_sale : ℝ := total_apples * (1 - first_day_sale_percentage)

/-- Represents the number of apples thrown away on the first day -/
def apples_thrown_first_day : ℝ := 0.2 * apples_after_first_sale

/-- Represents the number of apples remaining after throwing away on the first day -/
def apples_remaining_first_day : ℝ := apples_after_first_sale - apples_thrown_first_day

/-- Represents the number of apples sold on the second day -/
def apples_sold_second_day : ℝ := 0.5 * apples_remaining_first_day

/-- Represents the number of apples thrown away on the second day -/
def apples_thrown_second_day : ℝ := apples_remaining_first_day - apples_sold_second_day

/-- Represents the total number of apples thrown away -/
def total_apples_thrown : ℝ := apples_thrown_first_day + apples_thrown_second_day

theorem vendor_first_day_sale_percentage :
  first_day_sale_percentage = 0.5 ∧
  total_apples_thrown = 0.3 * total_apples :=
sorry

end vendor_first_day_sale_percentage_l1867_186767


namespace smallest_rectangle_containing_circle_l1867_186788

/-- The diameter of the circle -/
def circle_diameter : ℝ := 10

/-- The area of the smallest rectangle containing the circle -/
def smallest_rectangle_area : ℝ := 120

/-- Theorem stating that the area of the smallest rectangle containing a circle
    with diameter 10 units is 120 square units -/
theorem smallest_rectangle_containing_circle :
  smallest_rectangle_area = circle_diameter * (circle_diameter + 2) :=
by sorry

end smallest_rectangle_containing_circle_l1867_186788


namespace least_subtraction_for_common_remainder_l1867_186774

theorem least_subtraction_for_common_remainder (n : ℕ) : 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4) → 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4 ∧
   ∀ (y : ℕ), y < x → 
     ((642 - y) % 11 ≠ 4 ∨ 
      (642 - y) % 13 ≠ 4 ∨ 
      (642 - y) % 17 ≠ 4)) :=
by sorry

end least_subtraction_for_common_remainder_l1867_186774


namespace imaginary_part_of_z_l1867_186732

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end imaginary_part_of_z_l1867_186732


namespace smallest_value_3a_plus_2_l1867_186726

theorem smallest_value_3a_plus_2 (a : ℝ) (h : 4 * a^2 + 6 * a + 3 = 2) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (x : ℝ), (∃ (b : ℝ), 4 * b^2 + 6 * b + 3 = 2 ∧ x = 3 * b + 2) → x ≥ min :=
by sorry

end smallest_value_3a_plus_2_l1867_186726


namespace square_diagonal_length_l1867_186749

theorem square_diagonal_length (A : ℝ) (d : ℝ) : 
  A = 392 → d = 28 → d^2 = 2 * A :=
by
  sorry

end square_diagonal_length_l1867_186749


namespace hyperbola_min_focal_length_l1867_186724

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    focal length 2c, and a + b - c = 2, the minimum value of 2c is 4 + 4√2. -/
theorem hyperbola_min_focal_length (a b c : ℝ) : 
  a > 0 → b > 0 → a + b - c = 2 → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  2 * c ≥ 4 + 4 * Real.sqrt 2 := by
  sorry

end hyperbola_min_focal_length_l1867_186724


namespace kiwis_for_18_apples_l1867_186703

-- Define the costs of fruits in terms of an arbitrary unit
variable (apple_cost banana_cost cucumber_cost kiwi_cost : ℚ)

-- Define the conditions
axiom apple_banana_ratio : 9 * apple_cost = 3 * banana_cost
axiom banana_cucumber_ratio : banana_cost = 2 * cucumber_cost
axiom cucumber_kiwi_ratio : 3 * cucumber_cost = 4 * kiwi_cost

-- Define the theorem
theorem kiwis_for_18_apples : 
  ∃ n : ℕ, (18 * apple_cost = n * kiwi_cost) ∧ n = 16 := by
  sorry

end kiwis_for_18_apples_l1867_186703


namespace worker_idle_days_l1867_186733

theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_work_day : ℕ) 
  (deduction_per_idle_day : ℕ) 
  (total_payment : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_work_day = 20)
  (h3 : deduction_per_idle_day = 3)
  (h4 : total_payment = 280) :
  ∃ (idle_days : ℕ) (work_days : ℕ),
    idle_days + work_days = total_days ∧
    pay_per_work_day * work_days - deduction_per_idle_day * idle_days = total_payment ∧
    idle_days = 40 := by
  sorry

end worker_idle_days_l1867_186733


namespace gravitational_force_on_moon_l1867_186717

/-- Represents the gravitational force at a given distance from Earth's center -/
def gravitational_force (distance : ℝ) : ℝ := sorry

/-- The distance from Earth's center to its surface in miles -/
def earth_surface_distance : ℝ := 4000

/-- The distance from Earth's center to the moon in miles -/
def moon_distance : ℝ := 240000

/-- The gravitational force on Earth's surface in Newtons -/
def earth_surface_force : ℝ := 600

theorem gravitational_force_on_moon :
  gravitational_force earth_surface_distance = earth_surface_force →
  (∀ d : ℝ, gravitational_force d * d^2 = gravitational_force earth_surface_distance * earth_surface_distance^2) →
  gravitational_force moon_distance = 1/6 := by sorry

end gravitational_force_on_moon_l1867_186717


namespace tangent_circles_radii_relation_l1867_186778

/-- Three circles with centers O₁, O₂, and O₃, which are tangent to each other and a line -/
structure TangentCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ
  R₁ : ℝ
  R₂ : ℝ
  R₃ : ℝ
  tangent_to_line : Bool
  tangent_to_each_other : Bool

/-- The theorem stating the relationship between the radii of three tangent circles -/
theorem tangent_circles_radii_relation (tc : TangentCircles) :
  1 / Real.sqrt tc.R₂ = 1 / Real.sqrt tc.R₁ + 1 / Real.sqrt tc.R₃ :=
by sorry

end tangent_circles_radii_relation_l1867_186778


namespace melted_sphere_radius_l1867_186747

theorem melted_sphere_radius (r : ℝ) : 
  r > 0 → (4 / 3 * Real.pi * r^3 = 8 * (4 / 3 * Real.pi * 1^3)) → r = 2 := by
  sorry

end melted_sphere_radius_l1867_186747


namespace triangle_theorem_l1867_186741

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  t.a + t.c = 6 ∧
  (Real.sqrt 3 / 4) * (t.a^2 + t.c^2 - t.b^2) = (1/2) * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 := by
  sorry


end triangle_theorem_l1867_186741


namespace students_tree_planting_l1867_186716

/-- The number of apple trees planted by students -/
def apple_trees : ℕ := 47

/-- The number of orange trees planted by students -/
def orange_trees : ℕ := 27

/-- The total number of trees planted by students -/
def total_trees : ℕ := apple_trees + orange_trees

theorem students_tree_planting : total_trees = 74 := by
  sorry

end students_tree_planting_l1867_186716


namespace rainy_days_last_week_l1867_186794

theorem rainy_days_last_week (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) →
  (∃ (R : ℕ), R = 2) :=
sorry

end rainy_days_last_week_l1867_186794


namespace solution_set_of_inequality_l1867_186719

theorem solution_set_of_inequality (x : ℝ) :
  (x / (2 * x - 1) > 1) ↔ (1/2 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_l1867_186719


namespace arithmetic_calculations_l1867_186751

theorem arithmetic_calculations : 
  (78 - 14 * 2 = 50) ∧ 
  (500 - 296 - 104 = 100) ∧ 
  (360 - 300 / 5 = 300) ∧ 
  (84 / (16 / 4) = 21) := by
  sorry

end arithmetic_calculations_l1867_186751


namespace five_number_difference_l1867_186782

theorem five_number_difference (a b c d e : ℝ) 
  (h1 : (a + b + c + d) / 4 + e = 74)
  (h2 : (a + b + c + e) / 4 + d = 80)
  (h3 : (a + b + d + e) / 4 + c = 98)
  (h4 : (a + c + d + e) / 4 + b = 116)
  (h5 : (b + c + d + e) / 4 + a = 128) :
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 126 := by
  sorry

end five_number_difference_l1867_186782


namespace vector_subtraction_l1867_186725

/-- Given vectors BA and CA in ℝ², prove that BC = BA - CA -/
theorem vector_subtraction (BA CA : ℝ × ℝ) (h1 : BA = (2, 3)) (h2 : CA = (4, 7)) :
  (BA.1 - CA.1, BA.2 - CA.2) = (-2, -4) := by
  sorry

end vector_subtraction_l1867_186725


namespace cubic_equation_unique_solution_l1867_186729

theorem cubic_equation_unique_solution :
  ∃! (x y : ℕ+), (y : ℤ)^3 = (x : ℤ)^3 + 8*(x : ℤ)^2 - 6*(x : ℤ) + 8 ∧ x = 9 ∧ y = 11 := by
  sorry

end cubic_equation_unique_solution_l1867_186729


namespace total_popsicle_sticks_popsicle_sum_l1867_186799

theorem total_popsicle_sticks : ℕ → ℕ → ℕ → ℕ
  | gino_sticks, your_sticks, nick_sticks =>
    gino_sticks + your_sticks + nick_sticks

theorem popsicle_sum (gino_sticks your_sticks nick_sticks : ℕ) 
  (h1 : gino_sticks = 63)
  (h2 : your_sticks = 50)
  (h3 : nick_sticks = 82) :
  total_popsicle_sticks gino_sticks your_sticks nick_sticks = 195 :=
by
  sorry

end total_popsicle_sticks_popsicle_sum_l1867_186799


namespace correct_total_crayons_l1867_186750

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem correct_total_crayons : total_crayons = 53 := by
  sorry

end correct_total_crayons_l1867_186750


namespace triangle_area_with_angle_bisector_l1867_186789

/-- The area of a triangle given two sides and the angle bisector between them -/
theorem triangle_area_with_angle_bisector (a b l : ℝ) (ha : a > 0) (hb : b > 0) (hl : l > 0) :
  let area := l * (a + b) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - l^2 * (a + b)^2)
  ∃ (α : ℝ), α > 0 ∧ α < π/2 ∧
    (l * (a + b) / (2 * a * b) = Real.cos α) ∧
    area = (1/2) * a * b * Real.sin (2 * α) :=
by sorry

end triangle_area_with_angle_bisector_l1867_186789


namespace clock_synchronization_l1867_186776

/-- Represents the chiming behavior of a clock -/
structure Clock where
  strikes_per_hour : ℕ
  chime_rate : ℚ

/-- The scenario of the King's and Queen's clocks -/
def clock_scenario (h : ℕ) : Prop :=
  let king_clock : Clock := { strikes_per_hour := h, chime_rate := 3/2 }
  let queen_clock : Clock := { strikes_per_hour := h, chime_rate := 1 }
  (king_clock.chime_rate * queen_clock.strikes_per_hour : ℚ) + 2 = h

/-- The theorem stating that the synchronization occurs at 5 o'clock -/
theorem clock_synchronization : 
  clock_scenario 5 := by sorry

end clock_synchronization_l1867_186776


namespace quadratic_unique_solution_l1867_186755

theorem quadratic_unique_solution :
  ∃! (k x : ℚ), 5 * k * x^2 + 30 * x + 10 = 0 ∧ k = 9/2 ∧ x = -2/3 := by
  sorry

end quadratic_unique_solution_l1867_186755


namespace range_when_p_range_when_p_or_q_l1867_186713

/-- Proposition p: The range of the function y=log(x^2+2ax+2-a) is ℝ -/
def p (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*a*x + 2 - a)

/-- Proposition q: ∀x ∈ [0,1], x^2+2x+a ≥ 0 -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2*x + a ≥ 0

/-- If p is true, then a ≤ -2 or a ≥ 1 -/
theorem range_when_p (a : ℝ) : p a → a ≤ -2 ∨ a ≥ 1 := by
  sorry

/-- If p ∨ q is true, then a ≤ -2 or a ≥ 0 -/
theorem range_when_p_or_q (a : ℝ) : p a ∨ q a → a ≤ -2 ∨ a ≥ 0 := by
  sorry

end range_when_p_range_when_p_or_q_l1867_186713


namespace bisection_method_condition_l1867_186783

/-- A continuous function on a closed interval -/
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

/-- The bisection method is applicable on an interval -/
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

/-- Theorem: For the bisection method to be applicable on an interval [a, b],
    the function f must satisfy f(a) · f(b) < 0 -/
theorem bisection_method_condition (f : ℝ → ℝ) (a b : ℝ) :
  BisectionApplicable f a b → f a * f b < 0 := by
  sorry


end bisection_method_condition_l1867_186783


namespace max_value_cubic_ratio_l1867_186761

theorem max_value_cubic_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 ∧
  (x + y)^3 / (x^3 + y^3) = 4 ↔ x = y :=
by sorry

end max_value_cubic_ratio_l1867_186761


namespace no_distinct_roots_l1867_186727

theorem no_distinct_roots : ¬∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (a^2 - 2*b*a + c^2 = 0) ∧
  (b^2 - 2*c*b + a^2 = 0) ∧
  (c^2 - 2*a*c + b^2 = 0) := by
  sorry

end no_distinct_roots_l1867_186727


namespace max_min_f_l1867_186764

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 0 ∧ min = -Real.exp 1 := by
  sorry

end max_min_f_l1867_186764


namespace unique_number_with_properties_l1867_186736

theorem unique_number_with_properties : ∃! x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  (x * (x - 1) % 100000 = 0) ∧
  ((x / 1000) % 10 = 0) ∧
  ((x % 3125 = 0 ∧ (x - 1) % 32 = 0) ∨ ((x - 1) % 3125 = 0 ∧ x % 32 = 0)) :=
sorry

end unique_number_with_properties_l1867_186736


namespace spaghetti_to_manicotti_ratio_l1867_186787

/-- The ratio of students who preferred spaghetti to those who preferred manicotti -/
def pasta_preference_ratio (spaghetti_count manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 650

/-- The theorem stating the ratio of spaghetti preference to manicotti preference -/
theorem spaghetti_to_manicotti_ratio : 
  pasta_preference_ratio 250 100 = 5/2 := by
  sorry

end spaghetti_to_manicotti_ratio_l1867_186787


namespace triangle_height_to_bc_l1867_186797

/-- In a triangle ABC, given side lengths and an angle, prove the height to a specific side. -/
theorem triangle_height_to_bc (a b c h : ℝ) (B : ℝ) : 
  a = 2 → 
  b = Real.sqrt 7 → 
  B = π / 3 →
  c^2 = a^2 + b^2 - 2*a*c*(Real.cos B) →
  h = (a * c * Real.sin B) / a →
  h = 3 * Real.sqrt 3 / 2 := by
  sorry

end triangle_height_to_bc_l1867_186797


namespace no_solution_l1867_186743

-- Define the system of equations
def system (x₁ x₂ x₃ : ℝ) : Prop :=
  (x₁ + 4*x₂ + 10*x₃ = 1) ∧
  (0*x₁ - 5*x₂ - 13*x₃ = -1.25) ∧
  (0*x₁ + 0*x₂ + 0*x₃ = 1.25)

-- Theorem stating that the system has no solution
theorem no_solution : ¬∃ (x₁ x₂ x₃ : ℝ), system x₁ x₂ x₃ := by
  sorry

end no_solution_l1867_186743


namespace sum_of_fractions_equals_25_l1867_186705

-- Define functions to convert numbers from different bases to base 10
def base8ToBase10 (n : ℕ) : ℕ := sorry

def base4ToBase10 (n : ℕ) : ℕ := sorry

def base5ToBase10 (n : ℕ) : ℕ := sorry

def base3ToBase10 (n : ℕ) : ℕ := sorry

-- Define the numbers in their respective bases
def num1 : ℕ := 254  -- in base 8
def den1 : ℕ := 14   -- in base 4
def num2 : ℕ := 132  -- in base 5
def den2 : ℕ := 26   -- in base 3

-- Theorem to prove
theorem sum_of_fractions_equals_25 :
  (base8ToBase10 num1 / base4ToBase10 den1) + (base5ToBase10 num2 / base3ToBase10 den2) = 25 := by
  sorry

end sum_of_fractions_equals_25_l1867_186705


namespace windows_per_floor_l1867_186723

theorem windows_per_floor (floors : ℕ) (payment_per_window : ℚ) 
  (deduction_per_3days : ℚ) (days_taken : ℕ) (final_payment : ℚ) :
  floors = 3 →
  payment_per_window = 2 →
  deduction_per_3days = 1 →
  days_taken = 6 →
  final_payment = 16 →
  ∃ (windows_per_floor : ℕ), 
    windows_per_floor = 3 ∧
    (floors * windows_per_floor * payment_per_window - 
      (days_taken / 3 : ℚ) * deduction_per_3days = final_payment) :=
by
  sorry

end windows_per_floor_l1867_186723


namespace trig_identity_l1867_186728

theorem trig_identity : 
  Real.sin (44 * π / 180) * Real.cos (14 * π / 180) - 
  Real.cos (44 * π / 180) * Real.cos (76 * π / 180) = 1/2 := by
  sorry

end trig_identity_l1867_186728


namespace arithmetic_progression_quadratic_roots_l1867_186756

/-- Given non-zero real numbers a, b, c forming an arithmetic progression with b as the middle term,
    the quadratic equation ax^2 + 2√2bx + c = 0 has two distinct real roots. -/
theorem arithmetic_progression_quadratic_roots (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0 :=
by sorry

end arithmetic_progression_quadratic_roots_l1867_186756


namespace min_value_expression_l1867_186766

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a > b) (hbc : b > c) : 
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end min_value_expression_l1867_186766


namespace merchant_profit_l1867_186763

theorem merchant_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 20 → 
  discount_percent = 10 → 
  let marked_price := cost * (1 + markup_percent / 100)
  let final_price := marked_price * (1 - discount_percent / 100)
  let profit_percent := (final_price - cost) / cost * 100
  profit_percent = 8 := by
sorry

end merchant_profit_l1867_186763


namespace smallest_five_digit_multiple_of_18_l1867_186701

theorem smallest_five_digit_multiple_of_18 : ∀ n : ℕ, 
  n ≥ 10000 ∧ n ≤ 99999 ∧ n % 18 = 0 → n ≥ 10008 :=
by
  sorry

end smallest_five_digit_multiple_of_18_l1867_186701


namespace laurent_series_expansion_l1867_186745

/-- The Laurent series expansion of f(z) = 1 / (z^2 - 1)^2 in the region 0 < |z-1| < 2 -/
theorem laurent_series_expansion (z : ℂ) (h : 0 < Complex.abs (z - 1) ∧ Complex.abs (z - 1) < 2) :
  (fun z => 1 / (z^2 - 1)^2) z = ∑' n, ((-1)^n * (n + 3) / 2^(n + 4)) * (z - 1)^n :=
sorry

end laurent_series_expansion_l1867_186745


namespace sand_bag_fraction_l1867_186758

/-- Given a bag of sand weighing 50 kg, prove that after using 30 kg,
    the remaining sand accounts for 2/5 of the total bag. -/
theorem sand_bag_fraction (total_weight : ℝ) (used_weight : ℝ) 
  (h1 : total_weight = 50)
  (h2 : used_weight = 30) :
  (total_weight - used_weight) / total_weight = 2 / 5 := by
  sorry

end sand_bag_fraction_l1867_186758


namespace dot_product_range_l1867_186791

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse x^2 + y^2/9 = 1 -/
def isOnEllipse (p : Point) : Prop :=
  p.x^2 + p.y^2/9 = 1

/-- Checks if two points are symmetric about the origin -/
def areSymmetric (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- Calculates the dot product of vectors CA and CB -/
def dotProduct (a b c : Point) : ℝ :=
  (a.x - c.x) * (b.x - c.x) + (a.y - c.y) * (b.y - c.y)

theorem dot_product_range :
  ∀ (a b : Point),
    isOnEllipse a →
    isOnEllipse b →
    areSymmetric a b →
    let c := Point.mk 5 5
    41 ≤ dotProduct a b c ∧ dotProduct a b c ≤ 49 := by
  sorry


end dot_product_range_l1867_186791


namespace membership_condition_l1867_186734

def is_necessary_but_not_sufficient {α : Type*} (A B : Set α) : Prop :=
  (A ∩ B = B) ∧ (A ≠ B) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B)

theorem membership_condition {α : Type*} (A B : Set α) 
  (h1 : A ∩ B = B) (h2 : A ≠ B) :
  is_necessary_but_not_sufficient A B :=
sorry

end membership_condition_l1867_186734


namespace habitable_earth_surface_fraction_l1867_186769

theorem habitable_earth_surface_fraction :
  let total_surface : ℚ := 1
  let water_covered_fraction : ℚ := 2/3
  let land_fraction : ℚ := 1 - water_covered_fraction
  let inhabitable_land_fraction : ℚ := 2/3
  inhabitable_land_fraction * land_fraction = 2/9 :=
by sorry

end habitable_earth_surface_fraction_l1867_186769


namespace exists_natural_sqrt_nested_root_l1867_186780

theorem exists_natural_sqrt_nested_root : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (n : ℝ)^(5/4) = m := by
  sorry

end exists_natural_sqrt_nested_root_l1867_186780


namespace focus_of_specific_parabola_l1867_186714

/-- A parabola is defined by the equation y^2 = -4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -4 * p.1}

/-- The focus of a parabola is a point from which all points on the parabola are equidistant -/
def FocusOfParabola (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem focus_of_specific_parabola :
  FocusOfParabola Parabola = (-1, 0) := by sorry

end focus_of_specific_parabola_l1867_186714


namespace sqrt_3_simplest_l1867_186762

-- Define a function to represent the concept of simplicity for square roots
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (x = Real.sqrt y) → ¬∃ z : ℝ, z ≠ y ∧ Real.sqrt z = Real.sqrt y

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_sqrt (Real.sqrt 3) ∧
  ¬is_simplest_sqrt (Real.sqrt (a^2)) ∧
  ¬is_simplest_sqrt (Real.sqrt 0.3) ∧
  ¬is_simplest_sqrt (Real.sqrt 27) :=
sorry

end sqrt_3_simplest_l1867_186762


namespace rectangle_reassembly_l1867_186721

/-- For any positive real numbers a and b, there exists a way to cut and reassemble
    a rectangle with dimensions a and b into a rectangle with one side equal to 1. -/
theorem rectangle_reassembly (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (c : ℝ), c > 0 ∧ a * b = c * 1 := by
  sorry

end rectangle_reassembly_l1867_186721


namespace cube_surface_area_l1867_186718

theorem cube_surface_area (volume : ℝ) (surface_area : ℝ) : 
  volume = 3375 → surface_area = 1350 → 
  (∃ (side : ℝ), volume = side^3 ∧ surface_area = 6 * side^2) :=
by sorry

end cube_surface_area_l1867_186718


namespace quadratic_equation_roots_l1867_186753

theorem quadratic_equation_roots (k : ℝ) : 
  (2 : ℝ) ^ 2 + k * 2 - 10 = 0 → k = 3 ∧ ∃ x : ℝ, x ≠ 2 ∧ x ^ 2 + k * x - 10 = 0 ∧ x = -5 :=
by
  sorry

end quadratic_equation_roots_l1867_186753


namespace vector_arithmetic_l1867_186784

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the expected result. -/
theorem vector_arithmetic (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end vector_arithmetic_l1867_186784


namespace triangle_area_l1867_186707

/-- The area of a triangle with base 2 and height 3 is 3 -/
theorem triangle_area : 
  let base : ℝ := 2
  let height : ℝ := 3
  let area := (base * height) / 2
  area = 3 := by sorry

end triangle_area_l1867_186707


namespace lcm_9_12_15_l1867_186772

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_9_12_15_l1867_186772


namespace farm_hens_count_l1867_186779

/-- Given a farm with roosters and hens, proves that the number of hens is 67 -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by sorry

end farm_hens_count_l1867_186779


namespace cos_negative_45_degrees_l1867_186704

theorem cos_negative_45_degrees : Real.cos (-(Real.pi / 4)) = Real.sqrt 2 / 2 := by
  sorry

end cos_negative_45_degrees_l1867_186704


namespace completing_square_equivalence_l1867_186781

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end completing_square_equivalence_l1867_186781
