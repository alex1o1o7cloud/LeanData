import Mathlib

namespace fifth_term_is_13_l3848_384861

/-- A sequence where the difference between consecutive terms increases by 1 each time -/
def increasing_diff_seq (a₁ : ℕ) (d₁ : ℕ) : ℕ → ℕ
| 0 => a₁
| n + 1 => increasing_diff_seq a₁ d₁ n + d₁ + n

theorem fifth_term_is_13 (a₁ d₁ : ℕ) :
  a₁ = 3 ∧ d₁ = 1 →
  increasing_diff_seq a₁ d₁ 1 = 4 ∧
  increasing_diff_seq a₁ d₁ 2 = 6 ∧
  increasing_diff_seq a₁ d₁ 3 = 9 →
  increasing_diff_seq a₁ d₁ 4 = 13 := by
  sorry

#eval increasing_diff_seq 3 1 4  -- Should output 13

end fifth_term_is_13_l3848_384861


namespace largest_angle_in_special_triangle_l3848_384887

theorem largest_angle_in_special_triangle (α β γ : Real) : 
  α + β + γ = π ∧ 
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  Real.tan α + Real.tan β + Real.tan γ = 2016 →
  (max α (max β γ)) > π/2 - π/360 :=
by sorry

end largest_angle_in_special_triangle_l3848_384887


namespace marbles_distribution_l3848_384882

/-- The number of marbles distributed per class -/
def marbles_per_class : ℕ := 37

/-- The number of classes -/
def number_of_classes : ℕ := 23

/-- The number of leftover marbles -/
def leftover_marbles : ℕ := 16

/-- The total number of marbles distributed to students -/
def total_marbles : ℕ := marbles_per_class * number_of_classes + leftover_marbles

theorem marbles_distribution :
  total_marbles = 867 := by sorry

end marbles_distribution_l3848_384882


namespace imaginary_part_of_complex_fraction_l3848_384820

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by
sorry

end imaginary_part_of_complex_fraction_l3848_384820


namespace tickets_per_box_l3848_384812

theorem tickets_per_box (total_tickets : ℕ) (num_boxes : ℕ) (h1 : total_tickets = 45) (h2 : num_boxes = 9) :
  total_tickets / num_boxes = 5 := by
  sorry

end tickets_per_box_l3848_384812


namespace sam_average_letters_per_day_l3848_384871

/-- Given that Sam wrote 7 letters on Tuesday and 3 letters on Wednesday,
    prove that the average number of letters he wrote per day is 5. -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  let average_letters : ℚ := total_letters / total_days
  average_letters = 5 := by
sorry

end sam_average_letters_per_day_l3848_384871


namespace cube_surface_area_l3848_384853

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1331 →
  volume = side ^ 3 →
  surface_area = 6 * side ^ 2 →
  surface_area = 726 := by
sorry

end cube_surface_area_l3848_384853


namespace perpendicular_vectors_l3848_384805

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a i) * ((a i) - (b x i)) = 0) → x = 9 := by
  sorry

end perpendicular_vectors_l3848_384805


namespace max_area_rectangle_with_perimeter_60_l3848_384890

/-- The maximum area of a rectangle with perimeter 60 is 225 -/
theorem max_area_rectangle_with_perimeter_60 :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  2 * a + 2 * b = 60 →
  a * b ≤ 225 :=
by sorry

end max_area_rectangle_with_perimeter_60_l3848_384890


namespace quadratic_expression_value_l3848_384835

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 11) 
  (eq2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := by
sorry

end quadratic_expression_value_l3848_384835


namespace quadratic_polynomial_with_complex_root_l3848_384840

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
    (a * (4 + I)^2 + b * (4 + I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 51) :=
by sorry

end quadratic_polynomial_with_complex_root_l3848_384840


namespace fourth_root_sum_of_fourth_powers_l3848_384824

/-- Given segments a and b, there exists a segment x such that x^4 = a^4 + b^4 -/
theorem fourth_root_sum_of_fourth_powers (a b : ℝ) : ∃ x : ℝ, x^4 = a^4 + b^4 := by
  sorry

end fourth_root_sum_of_fourth_powers_l3848_384824


namespace pure_imaginary_complex_l3848_384897

def i : ℂ := Complex.I

theorem pure_imaginary_complex (a : ℝ) : 
  (∃ (b : ℝ), (2 - i) * (a - i) = b * i ∧ b ≠ 0) → a = 1/2 := by
  sorry

end pure_imaginary_complex_l3848_384897


namespace point_A_in_third_quadrant_l3848_384802

/-- A point in the Cartesian coordinate system is in the third quadrant if and only if
    both its x and y coordinates are negative. -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point A with coordinates (-1, -3) lies in the third quadrant. -/
theorem point_A_in_third_quadrant :
  third_quadrant (-1) (-3) := by
  sorry

end point_A_in_third_quadrant_l3848_384802


namespace max_tuesdays_in_63_days_l3848_384804

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we are considering -/
def total_days : ℕ := 63

/-- Each week has one Tuesday -/
axiom one_tuesday_per_week : ℕ

/-- The maximum number of Tuesdays in the first 63 days of a year -/
def max_tuesdays : ℕ := total_days / days_in_week

theorem max_tuesdays_in_63_days : max_tuesdays = 9 := by
  sorry

end max_tuesdays_in_63_days_l3848_384804


namespace library_wall_length_proof_l3848_384836

/-- The length of the library wall given the specified conditions -/
def library_wall_length : ℝ := 8

/-- Represents the number of desks (which is equal to the number of bookcases) -/
def num_furniture : ℕ := 2

theorem library_wall_length_proof :
  (∃ n : ℕ, 
    n = num_furniture ∧ 
    2 * n + 1.5 * n + 1 = library_wall_length ∧
    ∀ m : ℕ, m > n → 2 * m + 1.5 * m + 1 > library_wall_length) := by
  sorry

#check library_wall_length_proof

end library_wall_length_proof_l3848_384836


namespace word_to_number_correct_l3848_384825

def word_to_number (s : String) : ℝ :=
  match s with
  | "fifty point zero zero one" => 50.001
  | "seventy-five point zero six" => 75.06
  | _ => 0  -- Default case for other inputs

theorem word_to_number_correct :
  (word_to_number "fifty point zero zero one" = 50.001) ∧
  (word_to_number "seventy-five point zero six" = 75.06) := by
  sorry

end word_to_number_correct_l3848_384825


namespace total_seashells_l3848_384874

theorem total_seashells (sally tom jessica alex : ℝ) 
  (h1 : sally = 9.5)
  (h2 : tom = 7.2)
  (h3 : jessica = 5.3)
  (h4 : alex = 12.8) :
  sally + tom + jessica + alex = 34.8 := by
  sorry

end total_seashells_l3848_384874


namespace symbol_values_l3848_384864

theorem symbol_values (triangle star : ℤ) 
  (eq1 : 3 * triangle + 2 * star = 14)
  (eq2 : 2 * star + 5 * triangle = 18) : 
  triangle = 2 ∧ star = 4 := by
sorry

end symbol_values_l3848_384864


namespace probability_one_instrument_l3848_384855

theorem probability_one_instrument (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 1/5 →
  two_or_more = 128 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 1/25 :=
by sorry

end probability_one_instrument_l3848_384855


namespace smallest_n_with_common_factor_l3848_384881

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 14 → ¬(has_common_factor_greater_than_one (8*n + 3) (10*n - 4))) ∧
  (has_common_factor_greater_than_one (8*14 + 3) (10*14 - 4)) :=
sorry

end smallest_n_with_common_factor_l3848_384881


namespace brilliant_permutations_l3848_384856

def word := "BRILLIANT"

/-- The number of permutations of the letters in 'BRILLIANT' where no two adjacent letters are the same -/
def valid_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) -
  (Nat.factorial 8 / Nat.factorial 2 +
   Nat.factorial 8 / Nat.factorial 2 -
   Nat.factorial 7)

theorem brilliant_permutations :
  valid_permutations = 55440 :=
sorry

end brilliant_permutations_l3848_384856


namespace unique_solution_inequality_l3848_384828

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 4*a

theorem unique_solution_inequality (a : ℝ) : 
  (∃! x, |f a x| ≤ 2) ↔ a = -1 := by sorry

end unique_solution_inequality_l3848_384828


namespace number_problem_l3848_384877

theorem number_problem (x : ℚ) : (3 / 4) * x = x - 19 → x = 76 := by
  sorry

end number_problem_l3848_384877


namespace max_x_plus_y_l3848_384842

theorem max_x_plus_y (x y : ℝ) (h : x^2 + 3*y^2 = 1) :
  ∃ (max_x max_y : ℝ), max_x^2 + 3*max_y^2 = 1 ∧
  ∀ (a b : ℝ), a^2 + 3*b^2 = 1 → a + b ≤ max_x + max_y ∧
  max_x = Real.sqrt 3 / 2 := by
sorry

end max_x_plus_y_l3848_384842


namespace least_positive_integer_to_make_multiple_of_five_l3848_384893

theorem least_positive_integer_to_make_multiple_of_five (n : ℕ) : 
  (∃ k : ℕ, (789 + n) = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, (789 + m) = 5 * k) → n = 1 := by
  sorry

end least_positive_integer_to_make_multiple_of_five_l3848_384893


namespace checkerboard_coverage_unsolvable_boards_l3848_384845

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (removed_squares : ℕ)

/-- Determines if a checkerboard can be completely covered by dominoes -/
def can_cover (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x7 board -/
def board_5x7 : Checkerboard := ⟨5, 7, 0⟩

/-- 7x3 board with two removed squares -/
def board_7x3_modified : Checkerboard := ⟨7, 3, 2⟩

theorem unsolvable_boards :
  ¬(can_cover board_5x7) ∧ ¬(can_cover board_7x3_modified) := by
  sorry

end checkerboard_coverage_unsolvable_boards_l3848_384845


namespace carly_lollipop_ratio_l3848_384859

/-- Given a total number of lollipops and the number of grape lollipops,
    calculate the ratio of cherry lollipops to the total number of lollipops. -/
def lollipop_ratio (total : ℕ) (grape : ℕ) : ℚ :=
  let other_flavors := grape * 3
  let cherry := total - other_flavors
  (cherry : ℚ) / total

/-- Theorem stating that given the conditions in the problem,
    the ratio of cherry lollipops to the total is 1/2. -/
theorem carly_lollipop_ratio :
  lollipop_ratio 42 7 = 1 / 2 := by
  sorry

end carly_lollipop_ratio_l3848_384859


namespace quadratic_inequality_range_l3848_384858

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + (1/4 : ℝ) ≤ 0) ↔ (0 < a ∧ a < 4) :=
by sorry

end quadratic_inequality_range_l3848_384858


namespace at_op_four_six_l3848_384851

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 2 * a^2 - 2 * b^2

-- Theorem statement
theorem at_op_four_six : at_op 4 6 = -40 := by sorry

end at_op_four_six_l3848_384851


namespace quadratic_roots_range_l3848_384862

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2*x + 1 = 0 ∧ m * y^2 + 2*y + 1 = 0) ↔ 
  (m ≤ 1 ∧ m ≠ 0) :=
sorry

end quadratic_roots_range_l3848_384862


namespace abs_two_set_l3848_384827

theorem abs_two_set : {x : ℝ | |x| = 2} = {-2, 2} := by
  sorry

end abs_two_set_l3848_384827


namespace yangtze_farm_grass_consumption_l3848_384866

/-- Represents the grass consumption scenario on Yangtze Farm -/
structure GrassConsumption where
  /-- The amount of grass one cow eats in one day -/
  b : ℝ
  /-- The initial amount of grass -/
  g : ℝ
  /-- The rate of grass growth per day -/
  r : ℝ

/-- Given the conditions, proves that 36 cows will eat the grass in 3 days -/
theorem yangtze_farm_grass_consumption (gc : GrassConsumption) 
  (h1 : gc.g + 6 * gc.r = 24 * 6 * gc.b)  -- 24 cows eat the grass in 6 days
  (h2 : gc.g + 8 * gc.r = 21 * 8 * gc.b)  -- 21 cows eat the grass in 8 days
  : gc.g + 3 * gc.r = 36 * 3 * gc.b := by
  sorry


end yangtze_farm_grass_consumption_l3848_384866


namespace heather_biking_speed_l3848_384898

def total_distance : ℝ := 320
def num_days : ℝ := 8.0

theorem heather_biking_speed : total_distance / num_days = 40 := by
  sorry

end heather_biking_speed_l3848_384898


namespace second_triangle_side_length_l3848_384870

/-- Given a sequence of equilateral triangles where each triangle is formed by joining
    the midpoints of the sides of the previous triangle, if the first triangle has sides
    of 80 cm and the sum of all triangle perimeters is 480 cm, then the side length of
    the second triangle is 40 cm. -/
theorem second_triangle_side_length
  (first_triangle_side : ℝ)
  (total_perimeter : ℝ)
  (h1 : first_triangle_side = 80)
  (h2 : total_perimeter = 480)
  (h3 : total_perimeter = (3 * first_triangle_side) / (1 - 1/2)) :
  first_triangle_side / 2 = 40 :=
sorry

end second_triangle_side_length_l3848_384870


namespace square_area_ratio_l3848_384813

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((s * Real.sqrt 5)^2) = 1/5 := by
  sorry

end square_area_ratio_l3848_384813


namespace smallest_symmetric_set_l3848_384884

-- Define a point in the xy-plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set T
def T : Set Point := sorry

-- Define symmetry conditions
def symmetricAboutOrigin (p : Point) : Prop :=
  Point.mk (-p.x) (-p.y) ∈ T

def symmetricAboutXAxis (p : Point) : Prop :=
  Point.mk p.x (-p.y) ∈ T

def symmetricAboutYAxis (p : Point) : Prop :=
  Point.mk (-p.x) p.y ∈ T

def symmetricAboutNegativeDiagonal (p : Point) : Prop :=
  Point.mk (-p.y) (-p.x) ∈ T

-- State the theorem
theorem smallest_symmetric_set :
  (∀ p ∈ T, symmetricAboutOrigin p ∧ 
            symmetricAboutXAxis p ∧ 
            symmetricAboutYAxis p ∧ 
            symmetricAboutNegativeDiagonal p) →
  Point.mk 1 4 ∈ T →
  (∃ (s : Finset Point), s.card = 8 ∧ ↑s = T) ∧
  ¬∃ (s : Finset Point), s.card < 8 ∧ ↑s = T :=
by sorry

end smallest_symmetric_set_l3848_384884


namespace cows_for_96_days_l3848_384875

/-- Represents the number of cows that can eat all the grass in a given number of days -/
structure GrazingScenario where
  cows : ℕ
  days : ℕ

/-- Represents the meadow with growing grass -/
structure Meadow where
  scenario1 : GrazingScenario
  scenario2 : GrazingScenario
  growth_rate : ℚ

/-- Calculate the number of cows that can eat all the grass in 96 days -/
def calculate_cows (m : Meadow) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem cows_for_96_days (m : Meadow) : 
  m.scenario1 = ⟨70, 24⟩ → 
  m.scenario2 = ⟨30, 60⟩ → 
  calculate_cows m = 20 := by
  sorry

end cows_for_96_days_l3848_384875


namespace log_inequality_l3848_384839

theorem log_inequality (a b c : ℝ) (h1 : a < b) (h2 : 0 < c) (h3 : c < 1) :
  a * Real.log c > b * Real.log c := by
  sorry

end log_inequality_l3848_384839


namespace smallest_n_for_monochromatic_subgraph_l3848_384803

/-- A simple graph with 10 vertices and n edges, where edges are colored in two colors -/
structure ColoredGraph (n : ℕ) :=
  (edges : Fin n → Fin 10 × Fin 10)
  (color : Fin n → Bool)
  (simple : ∀ i : Fin n, (edges i).1 ≠ (edges i).2)

/-- A monochromatic triangle in a colored graph -/
def has_monochromatic_triangle (G : ColoredGraph n) : Prop :=
  ∃ (i j k : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges j ≠ G.edges k ∧
    G.color i = G.color j ∧ G.color j = G.color k

/-- A monochromatic quadrilateral in a colored graph -/
def has_monochromatic_quadrilateral (G : ColoredGraph n) : Prop :=
  ∃ (i j k l : Fin n), 
    G.edges i ≠ G.edges j ∧ G.edges i ≠ G.edges k ∧ G.edges i ≠ G.edges l ∧ 
    G.edges j ≠ G.edges k ∧ G.edges j ≠ G.edges l ∧ G.edges k ≠ G.edges l ∧
    G.color i = G.color j ∧ G.color j = G.color k ∧ G.color k = G.color l

/-- The main theorem -/
theorem smallest_n_for_monochromatic_subgraph : 
  (∀ G : ColoredGraph 31, has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G) ∧
  (∃ G : ColoredGraph 30, ¬(has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G)) :=
sorry

end smallest_n_for_monochromatic_subgraph_l3848_384803


namespace product_of_powers_equals_thousand_l3848_384879

theorem product_of_powers_equals_thousand :
  (10 ^ 0.25) * (10 ^ 0.25) * (10 ^ 0.5) * (10 ^ 0.5) * (10 ^ 0.75) * (10 ^ 0.75) = 1000 := by
  sorry

end product_of_powers_equals_thousand_l3848_384879


namespace divisibility_problem_l3848_384889

theorem divisibility_problem (a b c d m : ℤ) 
  (h_m_pos : m > 0)
  (h_ac : m ∣ a * c)
  (h_bd : m ∣ b * d)
  (h_sum : m ∣ b * c + a * d) :
  (m ∣ b * c) ∧ (m ∣ a * d) := by
  sorry

end divisibility_problem_l3848_384889


namespace specific_pyramid_side_edge_l3848_384826

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- The side edge length of a regular square pyramid -/
def side_edge_length (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem stating the side edge length of a specific regular square pyramid -/
theorem specific_pyramid_side_edge :
  let p : RegularSquarePyramid := ⟨4 * Real.sqrt 2, 32⟩
  side_edge_length p = 5 := by
  sorry

end specific_pyramid_side_edge_l3848_384826


namespace line_circle_intersection_l3848_384811

/-- A line in 2D space -/
structure Line where
  k : ℝ
  b : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a line and a circle intersect -/
def intersect (l : Line) (c : Circle) : Prop :=
  ∃ x y : ℝ, y = l.k * x + l.b ∧ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem line_circle_intersection (k : ℝ) :
  (∀ l : Line, l.b = 1 → intersect l (Circle.mk (0, 1) 1)) ∧
  (∃ l : Line, l.b ≠ 1 ∧ intersect l (Circle.mk (0, 1) 1)) :=
sorry

end line_circle_intersection_l3848_384811


namespace zero_in_interval_l3848_384848

def f (a x : ℝ) : ℝ := 3 * a * x - 1 - 2 * a

theorem zero_in_interval (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1) 1, f a x = 0) → 
  (a < -1/5 ∨ a > 1) := by
sorry

end zero_in_interval_l3848_384848


namespace rain_probability_both_days_l3848_384854

theorem rain_probability_both_days (prob_monday : ℝ) (prob_tuesday : ℝ) 
  (h1 : prob_monday = 0.4)
  (h2 : prob_tuesday = 0.3)
  (h3 : 0 ≤ prob_monday ∧ prob_monday ≤ 1)
  (h4 : 0 ≤ prob_tuesday ∧ prob_tuesday ≤ 1) :
  prob_monday * prob_tuesday = 0.12 :=
by
  sorry

end rain_probability_both_days_l3848_384854


namespace expression_value_l3848_384829

theorem expression_value (x y z : ℝ) 
  (hx : x = -5/4) 
  (hy : y = -3/2) 
  (hz : z = Real.sqrt 2) : 
  -2 * x^3 - y^2 + Real.sin z = 53/32 + Real.sin (Real.sqrt 2) := by
  sorry

end expression_value_l3848_384829


namespace complement_B_intersection_A_complement_B_l3848_384837

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem for the complement of B with respect to U
theorem complement_B : Set.compl B = {x : ℝ | x ≤ 1} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ Set.compl B = {x : ℝ | x < 0} := by sorry

end complement_B_intersection_A_complement_B_l3848_384837


namespace cosine_inequality_solution_l3848_384832

theorem cosine_inequality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end cosine_inequality_solution_l3848_384832


namespace range_of_f_l3848_384844

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end range_of_f_l3848_384844


namespace is_point_of_tangency_l3848_384886

/-- The point of tangency between two circles -/
def point_of_tangency : ℝ × ℝ := (2.5, 5)

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 17 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 49 = 0

/-- Theorem stating that point_of_tangency is the point of tangency between the two circles -/
theorem is_point_of_tangency :
  let (x, y) := point_of_tangency
  circle1 x y ∧ circle2 x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(circle1 x' y' ∧ circle2 x' y') :=
by sorry

end is_point_of_tangency_l3848_384886


namespace abs_difference_symmetry_l3848_384880

theorem abs_difference_symmetry (a b : ℚ) : |a - b| = |b - a| := by sorry

end abs_difference_symmetry_l3848_384880


namespace negation_of_exists_exponential_l3848_384806

theorem negation_of_exists_exponential (x : ℝ) :
  (¬ ∃ x₀ : ℝ, (2 : ℝ) ^ x₀ ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by
  sorry

end negation_of_exists_exponential_l3848_384806


namespace hyperbola_properties_l3848_384872

/-- A hyperbola with asymptotes y = ±2x passing through (1, 0) -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

theorem hyperbola_properties :
  ∀ (x y : ℝ),
    -- The equation represents a hyperbola
    (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
    -- With asymptotes y = ±2x
    (∃ (k : ℝ), k ≠ 0 ∧ (y = 2*x ∨ y = -2*x) → (x^2 - y^2/4 = k)) →
    -- Passing through the point (1, 0)
    hyperbola 1 0 →
    -- Then the hyperbola has the equation x² - y²/4 = 1
    hyperbola x y :=
by sorry

end hyperbola_properties_l3848_384872


namespace y_value_proof_l3848_384892

theorem y_value_proof (y : ℝ) (h : 9 / y^2 = 3 * y / 81) : y = 9 := by
  sorry

end y_value_proof_l3848_384892


namespace surface_area_of_sliced_solid_l3848_384810

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  X : ℝ × ℝ × ℝ
  Y : ℝ × ℝ × ℝ
  Z : ℝ × ℝ × ℝ

/-- The solid formed by slicing off a part of the prism -/
def SlicedSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Surface area of the sliced solid -/
def surfaceArea (s : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem surface_area_of_sliced_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.base_side = 10 →
  surfaceArea (SlicedSolid p m) = 100 + 25 * Real.sqrt 3 / 4 := by
  sorry

end surface_area_of_sliced_solid_l3848_384810


namespace direction_vector_of_line_l3848_384878

/-- Given a line l with equation x + y + 1 = 0, prove that (1, -1) is a direction vector of l. -/
theorem direction_vector_of_line (l : Set (ℝ × ℝ)) :
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + p.2 + 1 = 0) →
  ∃ t : ℝ, (1 + t, -1 + t) ∈ l := by sorry

end direction_vector_of_line_l3848_384878


namespace train_length_l3848_384883

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 42 →
  crossing_time = 60 →
  bridge_length = 200 →
  ∃ (train_length : ℝ), abs (train_length - 500.2) < 0.1 :=
by
  sorry


end train_length_l3848_384883


namespace gcd_282_470_l3848_384808

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by sorry

end gcd_282_470_l3848_384808


namespace sum_of_averages_equals_155_l3848_384816

def even_integers_to_100 : List ℕ := List.range 51 |> List.map (· * 2)
def even_integers_to_50 : List ℕ := List.range 26 |> List.map (· * 2)
def even_perfect_squares_to_250 : List ℕ := [0, 4, 16, 36, 64, 100, 144, 196]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem sum_of_averages_equals_155 :
  average even_integers_to_100 +
  average even_integers_to_50 +
  average even_perfect_squares_to_250 = 155 := by
  sorry

end sum_of_averages_equals_155_l3848_384816


namespace log_equation_solution_set_l3848_384873

theorem log_equation_solution_set :
  let S : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}
  ∀ x : ℝ, (Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)) ↔ x ∈ S :=
by sorry

end log_equation_solution_set_l3848_384873


namespace product_pricing_l3848_384814

theorem product_pricing (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 0.9 * a > b) : 
  0.9 * a - b = 0.2 * b := by
sorry

end product_pricing_l3848_384814


namespace spade_calculation_l3848_384888

/-- The spade operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : spade 5 (spade 7 8) = -200 := by sorry

end spade_calculation_l3848_384888


namespace smallest_integer_fraction_l3848_384894

theorem smallest_integer_fraction (y : ℤ) : (7 : ℚ) / 11 < (y : ℚ) / 17 ↔ 11 ≤ y := by sorry

end smallest_integer_fraction_l3848_384894


namespace factor_expression_l3848_384896

theorem factor_expression (b : ℝ) : 52 * b^2 + 208 * b = 52 * b * (b + 4) := by
  sorry

end factor_expression_l3848_384896


namespace no_integer_solutions_l3848_384860

theorem no_integer_solutions : ¬∃ (x y : ℤ), (x ≠ 0 ∧ y ≠ 0) ∧ (x^2 / y - y^2 / x = 3 * (2 + 1 / (x * y))) := by
  sorry

end no_integer_solutions_l3848_384860


namespace sum_of_xyz_l3848_384830

theorem sum_of_xyz (p q : ℝ) (x y z : ℤ) : 
  p^2 = 25/50 →
  q^2 = (3 + Real.sqrt 7)^2 / 14 →
  p < 0 →
  q > 0 →
  (p + q)^3 = (x : ℝ) * Real.sqrt (y : ℝ) / (z : ℝ) →
  x + y + z = 177230 := by
sorry

end sum_of_xyz_l3848_384830


namespace average_pen_price_l3848_384867

/-- Represents the types of pens --/
inductive PenType
  | A
  | B
  | C
  | D

/-- Given data about pen sales --/
def pen_data : List (PenType × Nat × Nat) :=
  [(PenType.A, 5, 5), (PenType.B, 3, 8), (PenType.C, 2, 27), (PenType.D, 1, 10)]

/-- Total number of pens sold --/
def total_pens : Nat := 50

/-- Theorem stating that the average unit price of pens sold is 2.26元 --/
theorem average_pen_price :
  let total_revenue := (pen_data.map (fun (_, price, quantity) => price * quantity)).sum
  let average_price := (total_revenue : ℚ) / total_pens
  average_price = 226 / 100 := by
  sorry

#check average_pen_price

end average_pen_price_l3848_384867


namespace fraction_to_decimal_l3848_384838

theorem fraction_to_decimal : (47 : ℚ) / (2^2 * 5^4) = 0.0188 := by sorry

end fraction_to_decimal_l3848_384838


namespace lcm_gcd_product_40_100_l3848_384834

theorem lcm_gcd_product_40_100 : Nat.lcm 40 100 * Nat.gcd 40 100 = 4000 := by
  sorry

end lcm_gcd_product_40_100_l3848_384834


namespace fixed_point_on_graph_l3848_384846

theorem fixed_point_on_graph (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 5 * k
  f 5 = 225 := by sorry

end fixed_point_on_graph_l3848_384846


namespace triangle_angle_B_l3848_384801

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  b = 2 * Real.sqrt 3 →
  C = 30 * π / 180 →
  (B = 60 * π / 180 ∨ B = 120 * π / 180) :=
by sorry

end triangle_angle_B_l3848_384801


namespace larger_integer_problem_l3848_384823

theorem larger_integer_problem (x y : ℤ) (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := by
  sorry

end larger_integer_problem_l3848_384823


namespace seokjin_position_l3848_384885

/-- Given the positions of Jungkook, Yoojeong, and Seokjin on the stairs,
    prove that Seokjin is 3 steps higher than Jungkook. -/
theorem seokjin_position (jungkook yoojeong seokjin : ℕ) 
  (h1 : jungkook = 19)
  (h2 : yoojeong = jungkook + 8)
  (h3 : seokjin = yoojeong - 5) :
  seokjin - jungkook = 3 := by
sorry

end seokjin_position_l3848_384885


namespace total_paving_cost_l3848_384876

/-- Represents a section of a room with its dimensions and slab cost -/
structure Section where
  length : ℝ
  width : ℝ
  slabCost : ℝ

/-- Calculates the cost of paving a section -/
def sectionCost (s : Section) : ℝ :=
  s.length * s.width * s.slabCost

/-- The three sections of the room -/
def sectionA : Section := { length := 8, width := 4.75, slabCost := 900 }
def sectionB : Section := { length := 6, width := 3.25, slabCost := 800 }
def sectionC : Section := { length := 5, width := 2.5, slabCost := 1000 }

/-- Theorem stating the total cost of paving the floor for the entire room -/
theorem total_paving_cost :
  sectionCost sectionA + sectionCost sectionB + sectionCost sectionC = 62300 := by
  sorry


end total_paving_cost_l3848_384876


namespace little_twelve_games_l3848_384807

/-- Represents a basketball conference with divisions and teams. -/
structure BasketballConference where
  num_divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of scheduled games in the conference. -/
def total_games (conf : BasketballConference) : ℕ :=
  let intra_games := conf.num_divisions * (conf.teams_per_division.choose 2) * conf.intra_division_games
  let inter_games := (conf.num_divisions * (conf.num_divisions - 1) / 2) * (conf.teams_per_division ^ 2) * conf.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the Little Twelve Basketball Conference has 102 scheduled games. -/
theorem little_twelve_games :
  let conf : BasketballConference := {
    num_divisions := 3,
    teams_per_division := 4,
    intra_division_games := 3,
    inter_division_games := 2
  }
  total_games conf = 102 := by
  sorry


end little_twelve_games_l3848_384807


namespace least_grood_number_l3848_384852

theorem least_grood_number (n : ℕ) : n ≥ 10 ↔ (n * (n + 1) : ℚ) / 4 > n^2 := by sorry

end least_grood_number_l3848_384852


namespace units_digit_sum_factorials_l3848_384819

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials :
  units_digit (sum_factorials 99) = units_digit (sum_factorials 4) :=
by sorry

end units_digit_sum_factorials_l3848_384819


namespace vector_expression_simplification_l3848_384800

variable {V : Type*} [AddCommGroup V]

theorem vector_expression_simplification
  (CE AC DE AD : V) :
  CE + AC - DE - AD = (0 : V) := by
  sorry

end vector_expression_simplification_l3848_384800


namespace sum_of_digits_of_greatest_prime_factor_l3848_384833

def number : Nat := 15999

-- Define a function to get the greatest prime factor
def greatest_prime_factor (n : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_factor :
  sum_of_digits (greatest_prime_factor number) = 17 := by
  sorry

end sum_of_digits_of_greatest_prime_factor_l3848_384833


namespace cubic_function_properties_monotonicity_interval_l3848_384843

/-- A cubic function f(x) = ax^3 + bx^2 passing through (1,4) with slope 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 4 ∧ f' a b 1 = 9 → a = 1 ∧ b = 3 :=
sorry

theorem monotonicity_interval (a b m : ℝ) :
  (a = 1 ∧ b = 3) →
  (∀ x ∈ Set.Icc m (m + 1), f' a b x ≥ 0) ↔ (m ≥ 0 ∨ m ≤ -3) :=
sorry

end cubic_function_properties_monotonicity_interval_l3848_384843


namespace museum_ticket_problem_l3848_384868

/-- Represents the cost calculation for museum tickets with discounts -/
structure TicketCost where
  basePrice : ℕ
  option1Discount : ℚ
  option2Discount : ℚ
  freeTickets : ℕ

/-- Calculates the cost for Option 1 -/
def option1Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option1Discount) * students

/-- Calculates the cost for Option 2 -/
def option2Cost (tc : TicketCost) (students : ℕ) : ℚ :=
  tc.basePrice * (1 - tc.option2Discount) * (students - tc.freeTickets)

theorem museum_ticket_problem (tc : TicketCost)
    (h1 : tc.basePrice = 30)
    (h2 : tc.option1Discount = 0.3)
    (h3 : tc.option2Discount = 0.2)
    (h4 : tc.freeTickets = 5) :
  (option1Cost tc 45 < option2Cost tc 45) ∧
  (∃ x : ℕ, x = 40 ∧ option1Cost tc x = option2Cost tc x) := by
  sorry


end museum_ticket_problem_l3848_384868


namespace waiter_tips_fraction_l3848_384822

theorem waiter_tips_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  tips = (5/3) * salary → 
  income = salary + tips → 
  tips / income = 5/8 := by
sorry

end waiter_tips_fraction_l3848_384822


namespace problem_statements_l3848_384817

theorem problem_statements :
  (∀ x : ℤ, x^2 + 1 > 0) ∧
  (∃ x y : ℝ, x + y > 5 ∧ ¬(x > 2 ∧ y > 3)) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ y : ℝ, y ≤ 3 → ∃ x : ℝ, y = -x^2 + 2*x + 2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 2 ≤ 3) :=
by sorry

end problem_statements_l3848_384817


namespace cubic_root_sum_squares_over_product_l3848_384821

theorem cubic_root_sum_squares_over_product (k : ℤ) (hk : k ≠ 0) 
  (a b c : ℂ) (h : ∀ x : ℂ, x^3 + 10*x^2 + 5*x - k = 0 ↔ x = a ∨ x = b ∨ x = c) : 
  (a^2 + b^2 + c^2) / (a * b * c) = 90 / k := by
sorry

end cubic_root_sum_squares_over_product_l3848_384821


namespace correct_new_balance_l3848_384850

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_expense towels_return : ℚ) : ℚ :=
  initial_balance + groceries_expense + (groceries_expense / 2) - towels_return

/-- Proves that the new balance is correct given the specified transactions -/
theorem correct_new_balance :
  new_balance 126 60 45 = 171 := by
  sorry

end correct_new_balance_l3848_384850


namespace impossible_arrangement_l3848_384847

/-- Represents a 3x3 grid filled with digits --/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of available digits --/
def AvailableDigits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Check if a list of cells contains three different digits --/
def hasThreeDifferentDigits (g : Grid) (cells : List (Fin 3 × Fin 3)) : Prop :=
  (cells.map (fun (i, j) => g i j)).toFinset.card = 3

/-- Check if all rows, columns, and diagonals have three different digits --/
def isValidArrangement (g : Grid) : Prop :=
  (∀ i : Fin 3, hasThreeDifferentDigits g [(i, 0), (i, 1), (i, 2)]) ∧
  (∀ j : Fin 3, hasThreeDifferentDigits g [(0, j), (1, j), (2, j)]) ∧
  hasThreeDifferentDigits g [(0, 0), (1, 1), (2, 2)] ∧
  hasThreeDifferentDigits g [(0, 2), (1, 1), (2, 0)]

/-- Main theorem: It's impossible to arrange the digits as described --/
theorem impossible_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry


end impossible_arrangement_l3848_384847


namespace peach_difference_l3848_384831

theorem peach_difference (jill steven jake : ℕ) : 
  jill = 12 →
  steven = jill + 15 →
  jake = steven - 16 →
  jill - jake = 1 := by
sorry

end peach_difference_l3848_384831


namespace scientists_sum_equals_total_germany_japan_us_ratio_l3848_384849

/-- The total number of scientists in the research project. -/
def total_scientists : ℕ := 150

/-- The number of scientists from Germany. -/
def germany_scientists : ℕ := 27

/-- The number of scientists from other European countries. -/
def other_europe_scientists : ℕ := 33

/-- The number of scientists from Japan. -/
def japan_scientists : ℕ := 18

/-- The number of scientists from China. -/
def china_scientists : ℕ := 15

/-- The number of scientists from other Asian countries. -/
def other_asia_scientists : ℕ := 12

/-- The number of scientists from Canada. -/
def canada_scientists : ℕ := 23

/-- The number of scientists from the United States. -/
def us_scientists : ℕ := 12

/-- The number of scientists from South America. -/
def south_america_scientists : ℕ := 8

/-- The number of scientists from Australia. -/
def australia_scientists : ℕ := 3

/-- Theorem stating that the sum of scientists from all countries equals the total number of scientists. -/
theorem scientists_sum_equals_total :
  germany_scientists + other_europe_scientists + japan_scientists + china_scientists +
  other_asia_scientists + canada_scientists + us_scientists + south_america_scientists +
  australia_scientists = total_scientists :=
by sorry

/-- Theorem stating the ratio of scientists from Germany, Japan, and the United States. -/
theorem germany_japan_us_ratio :
  ∃ (k : ℕ), k ≠ 0 ∧ germany_scientists = 9 * k ∧ japan_scientists = 6 * k ∧ us_scientists = 4 * k :=
by sorry

end scientists_sum_equals_total_germany_japan_us_ratio_l3848_384849


namespace power_two_geq_double_plus_two_l3848_384895

theorem power_two_geq_double_plus_two (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2*(n+1) := by
  sorry

end power_two_geq_double_plus_two_l3848_384895


namespace geometric_series_sum_l3848_384818

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S := ∑' n, a * r^n
  S = 4/3 := by
sorry

end geometric_series_sum_l3848_384818


namespace weeks_to_save_shirt_l3848_384815

/-- 
Given:
- shirt_cost: The cost of the shirt in dollars
- initial_savings: The amount Macey has already saved in dollars
- weekly_savings: The amount Macey saves per week in dollars

Prove that the number of weeks needed to save the remaining amount is 3.
-/
theorem weeks_to_save_shirt (shirt_cost initial_savings weekly_savings : ℚ) 
  (h1 : shirt_cost = 3)
  (h2 : initial_savings = 3/2)
  (h3 : weekly_savings = 1/2) :
  (shirt_cost - initial_savings) / weekly_savings = 3 := by
sorry

end weeks_to_save_shirt_l3848_384815


namespace tangent_line_m_values_l3848_384891

/-- The equation of a line that may be tangent to a circle -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y + 8 = 0

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (m : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation x y m ∧ circle_equation x y

/-- Theorem stating the possible values of m when the line is tangent to the circle -/
theorem tangent_line_m_values :
  ∀ m : ℝ, is_tangent m → m = -3 ∨ m = -13 := by sorry

end tangent_line_m_values_l3848_384891


namespace dinner_cakes_count_l3848_384899

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by sorry

end dinner_cakes_count_l3848_384899


namespace race_head_start_l3848_384857

/-- Given two runners A and B, where A's speed is 20/19 times B's speed,
    the head start fraction that A should give B for a dead heat is 1/20 of the race length. -/
theorem race_head_start (speedA speedB : ℝ) (length headStart : ℝ) :
  speedA = (20 / 19) * speedB →
  (length / speedA = (length - headStart) / speedB) →
  headStart = (1 / 20) * length :=
by sorry

end race_head_start_l3848_384857


namespace ellipse_tangent_properties_l3848_384863

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the companion circle E
def companion_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a point on the companion circle
def point_on_circle (P : ℝ × ℝ) : Prop := companion_circle P.1 P.2

-- Define a tangent line to the ellipse
def is_tangent (P A : ℝ × ℝ) : Prop :=
  point_on_circle P ∧ ellipse A.1 A.2 ∧
  ∀ t : ℝ, t ≠ 0 → ¬(ellipse (A.1 + t * (P.1 - A.1)) (A.2 + t * (P.2 - A.2)))

-- Main theorem
theorem ellipse_tangent_properties :
  ∀ P A B Q : ℝ × ℝ,
  point_on_circle P →
  is_tangent P A →
  is_tangent P B →
  companion_circle Q.1 Q.2 →
  (∃ t : ℝ, Q.1 = A.1 + t * (P.1 - A.1) ∧ Q.2 = A.2 + t * (P.2 - A.2)) →
  (A ≠ B) →
  (∀ k₁ k₂ : ℝ,
    (P.1 ≠ 0 ∨ P.2 ≠ 0) →
    (Q.1 ≠ 0 ∨ Q.2 ≠ 0) →
    k₁ = P.2 / P.1 →
    k₂ = Q.2 / Q.1 →
    (((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0) ∧
     (k₁ * k₂ = -1/3))) :=
sorry

end ellipse_tangent_properties_l3848_384863


namespace max_profit_theorem_l3848_384809

/-- Represents the unit prices and quantities of exercise books -/
structure BookPrices where
  regular : ℝ
  deluxe : ℝ

/-- Represents the purchase quantities of exercise books -/
structure PurchaseQuantities where
  regular : ℝ
  deluxe : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions (prices : BookPrices) : Prop :=
  150 * prices.regular + 100 * prices.deluxe = 1450 ∧
  200 * prices.regular + 50 * prices.deluxe = 1100

/-- Defines the profit function -/
def profit_function (prices : BookPrices) (quantities : PurchaseQuantities) : ℝ :=
  (prices.regular - 2) * quantities.regular + (prices.deluxe - 7) * quantities.deluxe

/-- Defines the purchase constraints -/
def purchase_constraints (quantities : PurchaseQuantities) : Prop :=
  quantities.regular + quantities.deluxe = 500 ∧
  quantities.regular ≥ 3 * quantities.deluxe

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem (prices : BookPrices) 
  (h_conditions : problem_conditions prices) :
  ∃ (quantities : PurchaseQuantities),
    purchase_constraints quantities ∧
    profit_function prices quantities = 750 ∧
    ∀ (other_quantities : PurchaseQuantities),
      purchase_constraints other_quantities →
      profit_function prices other_quantities ≤ 750 :=
sorry

end max_profit_theorem_l3848_384809


namespace triangle_inequality_l3848_384841

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end triangle_inequality_l3848_384841


namespace equivalent_representations_l3848_384869

theorem equivalent_representations (n : ℕ+) :
  (∃ (x y : ℕ+), n = 3 * x^2 + y^2) ↔ (∃ (u v : ℕ+), n = u^2 + u * v + v^2) :=
by sorry

end equivalent_representations_l3848_384869


namespace school_referendum_non_voters_l3848_384865

theorem school_referendum_non_voters (total : ℝ) (yes_votes : ℝ) (no_votes : ℝ)
  (h1 : yes_votes = (3 / 5) * total)
  (h2 : no_votes = 0.28 * total)
  (h3 : total > 0) :
  (total - (yes_votes + no_votes)) / total = 0.12 := by
  sorry

end school_referendum_non_voters_l3848_384865
