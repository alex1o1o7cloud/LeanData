import Mathlib

namespace fraction_filled_equals_half_l3745_374593

/-- Represents the fraction of a cistern that can be filled in 15 minutes -/
def fraction_filled_in_15_min : ℚ := 1 / 2

/-- The time it takes to fill half of the cistern -/
def time_to_fill_half : ℕ := 15

/-- Theorem stating that the fraction of the cistern filled in 15 minutes is 1/2 -/
theorem fraction_filled_equals_half : 
  fraction_filled_in_15_min = 1 / 2 :=
by sorry

end fraction_filled_equals_half_l3745_374593


namespace ratio_of_sums_l3745_374512

theorem ratio_of_sums (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_squares_pqr : p^2 + q^2 + r^2 = 49)
  (sum_squares_uvw : u^2 + v^2 + w^2 = 64)
  (sum_products : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_of_sums_l3745_374512


namespace coefficient_x_cubed_expansion_l3745_374554

theorem coefficient_x_cubed_expansion : 
  let f : Polynomial ℚ := (X - 1) * (2 * X + 1)^5
  (f.coeff 3) = -40 := by
  sorry

end coefficient_x_cubed_expansion_l3745_374554


namespace expression_evaluation_l3745_374592

theorem expression_evaluation : 
  let x := Real.sqrt (6000 - (3^3 : ℝ))
  let y := (105 / 21 : ℝ)^2
  abs (x * y - 1932.25) < 0.01 := by sorry

end expression_evaluation_l3745_374592


namespace pole_height_is_seven_meters_l3745_374597

/-- Represents the geometry of a leaning telephone pole supported by a cable --/
structure LeaningPole where
  /-- Angle between the pole and the horizontal ground in degrees --/
  angle : ℝ
  /-- Distance from the pole base to the cable attachment point on the ground in meters --/
  cable_ground_distance : ℝ
  /-- Height of the person touching the cable in meters --/
  person_height : ℝ
  /-- Distance the person walks from the pole base towards the cable attachment point in meters --/
  person_distance : ℝ

/-- Calculates the height of the leaning pole given the geometry --/
def calculate_pole_height (pole : LeaningPole) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the pole height is 7 meters --/
theorem pole_height_is_seven_meters (pole : LeaningPole) 
  (h_angle : pole.angle = 85)
  (h_cable : pole.cable_ground_distance = 4)
  (h_person_height : pole.person_height = 1.75)
  (h_person_distance : pole.person_distance = 3)
  : calculate_pole_height pole = 7 := by
  sorry

end pole_height_is_seven_meters_l3745_374597


namespace willy_tv_series_completion_time_l3745_374535

/-- The number of days required to finish a TV series -/
def days_to_finish (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Theorem: It takes 30 days to finish the TV series under given conditions -/
theorem willy_tv_series_completion_time :
  days_to_finish 3 20 2 = 30 := by
  sorry

end willy_tv_series_completion_time_l3745_374535


namespace midpoint_sum_equals_vertex_sum_l3745_374513

/-- Given a quadrilateral in the Cartesian plane, the sum of the x-coordinates
    of the midpoints of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_sum_equals_vertex_sum (p q r s : ℝ) :
  let vertex_sum := p + q + r + s
  let midpoint_sum := (p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2
  midpoint_sum = vertex_sum := by sorry

end midpoint_sum_equals_vertex_sum_l3745_374513


namespace two_children_gender_combinations_l3745_374546

-- Define the Gender type
inductive Gender
  | Male
  | Female

-- Define a type for a pair of children
def ChildPair := (Gender × Gender)

-- Define the set of all possible gender combinations
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

-- Theorem statement
theorem two_children_gender_combinations :
  ∀ (family : Set ChildPair),
  (∀ pair : ChildPair, pair ∈ family ↔ pair ∈ allGenderCombinations) ↔
  family = allGenderCombinations :=
by sorry

end two_children_gender_combinations_l3745_374546


namespace smallest_perfect_square_divisible_by_4_and_9_l3745_374507

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem smallest_perfect_square_divisible_by_4_and_9 :
  ∀ n : ℕ, n > 0 → is_perfect_square n → n % 4 = 0 → n % 9 = 0 → n ≥ 36 :=
sorry

end smallest_perfect_square_divisible_by_4_and_9_l3745_374507


namespace intersection_A_complement_B_l3745_374562

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 2, 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {-1, 0, 1} := by sorry

end intersection_A_complement_B_l3745_374562


namespace village_population_l3745_374576

theorem village_population (population_95_percent : ℝ) (h : population_95_percent = 57200) :
  ∃ total_population : ℕ, 
    (↑total_population : ℝ) ≥ population_95_percent / 0.95 ∧ 
    (↑total_population : ℝ) < population_95_percent / 0.95 + 1 ∧
    total_population = 60211 := by
  sorry

end village_population_l3745_374576


namespace motorcyclist_wait_time_l3745_374517

/-- Given a hiker and a motorcyclist with specified speeds, prove the time it takes for the
    motorcyclist to cover the distance the hiker walks in 48 minutes. -/
theorem motorcyclist_wait_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) 
    (hiker_walk_time : ℝ) (h1 : hiker_speed = 6) (h2 : motorcyclist_speed = 30) 
    (h3 : hiker_walk_time = 48) :
    (hiker_speed * hiker_walk_time) / motorcyclist_speed = 9.6 := by
  sorry

#check motorcyclist_wait_time

end motorcyclist_wait_time_l3745_374517


namespace property_one_property_two_property_three_f_satisfies_all_properties_l3745_374543

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Property 1: f(x₁x₂) = f(x₁)f(x₂)
theorem property_one : ∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂ := by sorry

-- Property 2: For x ∈ (0, +∞), f'(x) > 0
theorem property_two : ∀ x : ℝ, x > 0 → (deriv f) x > 0 := by sorry

-- Property 3: f'(x) is an odd function
theorem property_three : ∀ x : ℝ, (deriv f) (-x) = -(deriv f) x := by sorry

-- Main theorem: f(x) = x² satisfies all three properties
theorem f_satisfies_all_properties : 
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (∀ x : ℝ, x > 0 → (deriv f) x > 0) ∧ 
  (∀ x : ℝ, (deriv f) (-x) = -(deriv f) x) := by sorry

end property_one_property_two_property_three_f_satisfies_all_properties_l3745_374543


namespace eight_percent_of_1200_is_96_l3745_374552

theorem eight_percent_of_1200_is_96 : 
  (8 / 100) * 1200 = 96 := by sorry

end eight_percent_of_1200_is_96_l3745_374552


namespace garage_wheels_count_l3745_374590

/-- The number of wheels on a bicycle -/
def bicycle_wheels : Nat := 2

/-- The number of wheels on a car -/
def car_wheels : Nat := 4

/-- The number of wheels on a motorcycle -/
def motorcycle_wheels : Nat := 2

/-- The number of bicycles in the garage -/
def num_bicycles : Nat := 20

/-- The number of cars in the garage -/
def num_cars : Nat := 10

/-- The number of motorcycles in the garage -/
def num_motorcycles : Nat := 5

/-- The total number of wheels in the garage -/
def total_wheels : Nat := bicycle_wheels * num_bicycles + car_wheels * num_cars + motorcycle_wheels * num_motorcycles

theorem garage_wheels_count : total_wheels = 90 := by
  sorry

end garage_wheels_count_l3745_374590


namespace A_intersect_B_eq_A_l3745_374524

-- Define sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}

-- Theorem statement
theorem A_intersect_B_eq_A : A ∩ B = A := by sorry

end A_intersect_B_eq_A_l3745_374524


namespace justin_reading_ratio_l3745_374565

/-- Proves that the ratio of pages read each day in the remaining 6 days to the first day is 2:1 -/
theorem justin_reading_ratio : ∀ (pages_first_day : ℕ) (total_pages : ℕ) (days_remaining : ℕ),
  pages_first_day = 10 →
  total_pages = 130 →
  days_remaining = 6 →
  (days_remaining * (pages_first_day * (total_pages - pages_first_day) / (pages_first_day * days_remaining)) = total_pages - pages_first_day) →
  (total_pages - pages_first_day) / (pages_first_day * days_remaining) = 2 := by
  sorry

end justin_reading_ratio_l3745_374565


namespace perfect_square_m_l3745_374506

theorem perfect_square_m (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Odd k) (h5 : (2 + Real.sqrt 3)^k = 1 + m + n * Real.sqrt 3) : 
  ∃ (q : ℕ), m = q^2 := by
sorry

end perfect_square_m_l3745_374506


namespace marble_collection_total_l3745_374556

theorem marble_collection_total (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.3 * b → 
  g = 1.5 * r → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((r + b + g) / r - 3.27) < ε :=
sorry

end marble_collection_total_l3745_374556


namespace sum_of_series_l3745_374584

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series : ∑' k, (k : ℝ) / 3^k = 3/4 := by sorry

end sum_of_series_l3745_374584


namespace max_x_coordinate_P_max_x_coordinate_P_achieved_l3745_374572

/-- The maximum x-coordinate of point P on line OA, where A is on the ellipse x²/16 + y²/4 = 1 and OA · OP = 6 -/
theorem max_x_coordinate_P (A : ℝ × ℝ) (P : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  (∃ t : ℝ, P = (t * A.1, t * A.2)) →  -- P is on the line OA
  (A.1 * P.1 + A.2 * P.2 = 6) →  -- OA · OP = 6
  P.1 ≤ Real.sqrt 3 := by
sorry

/-- The maximum x-coordinate of point P is achieved -/
theorem max_x_coordinate_P_achieved (A : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  ∃ P : ℝ × ℝ, 
    (∃ t : ℝ, P = (t * A.1, t * A.2)) ∧  -- P is on the line OA
    (A.1 * P.1 + A.2 * P.2 = 6) ∧  -- OA · OP = 6
    P.1 = Real.sqrt 3 := by
sorry

end max_x_coordinate_P_max_x_coordinate_P_achieved_l3745_374572


namespace find_common_ratio_l3745_374545

/-- Given a table of n^2 (n ≥ 4) positive numbers arranged in n rows and n columns,
    where each row forms an arithmetic sequence and each column forms a geometric sequence
    with the same common ratio q, prove that q = 1/2 given the specified conditions. -/
theorem find_common_ratio (n : ℕ) (a : ℕ → ℕ → ℝ) (q : ℝ) 
    (h_n : n ≥ 4)
    (h_positive : ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i j > 0)
    (h_arithmetic_row : ∀ i k, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k < n → 
      a i (k + 1) - a i k = a i (k + 2) - a i (k + 1))
    (h_geometric_col : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j ≤ n → 
      a (i + 1) j = q * a i j)
    (h_a26 : a 2 6 = 1)
    (h_a42 : a 4 2 = 1/8)
    (h_a44 : a 4 4 = 3/16) :
  q = 1/2 := by
  sorry

end find_common_ratio_l3745_374545


namespace bridget_profit_l3745_374583

def total_loaves : ℕ := 60
def morning_price : ℚ := 3
def afternoon_price : ℚ := 2
def late_price : ℚ := 3/2
def production_cost : ℚ := 4/5

def morning_sales : ℕ := total_loaves / 3
def afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3) / 4
def late_sales : ℕ := total_loaves - morning_sales - afternoon_sales

def total_revenue : ℚ := 
  morning_sales * morning_price + 
  afternoon_sales * afternoon_price + 
  late_sales * late_price

def total_cost : ℚ := total_loaves * production_cost

def profit : ℚ := total_revenue - total_cost

theorem bridget_profit : profit = 87 := by sorry

end bridget_profit_l3745_374583


namespace probability_two_females_l3745_374532

/-- The probability of selecting 2 female students from a group of 5 students (2 males and 3 females) -/
theorem probability_two_females (total_students : Nat) (male_students : Nat) (female_students : Nat) 
  (group_size : Nat) : 
  total_students = 5 → 
  male_students = 2 → 
  female_students = 3 → 
  group_size = 2 → 
  (Nat.choose female_students group_size : Rat) / (Nat.choose total_students group_size : Rat) = 3/10 :=
by sorry

end probability_two_females_l3745_374532


namespace integer_triple_divisibility_l3745_374561

theorem integer_triple_divisibility : 
  {(a, b, c) : ℤ × ℤ × ℤ | 1 < a ∧ a < b ∧ b < c ∧ (abc - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0} = 
  {(3, 5, 15), (2, 4, 8)} := by sorry

end integer_triple_divisibility_l3745_374561


namespace remainder_problem_l3745_374501

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 := by
  sorry

end remainder_problem_l3745_374501


namespace min_toothpicks_to_remove_l3745_374566

/-- Represents a square lattice made of toothpicks -/
structure SquareLattice where
  size : Nat
  total_toothpicks : Nat
  boundary_toothpicks : Nat
  internal_grid_toothpicks : Nat
  diagonal_toothpicks : Nat

/-- Theorem: Minimum number of toothpicks to remove to eliminate all squares and triangles -/
theorem min_toothpicks_to_remove (lattice : SquareLattice) 
  (h1 : lattice.size = 3)
  (h2 : lattice.total_toothpicks = 40)
  (h3 : lattice.boundary_toothpicks = 12)
  (h4 : lattice.internal_grid_toothpicks = 4)
  (h5 : lattice.diagonal_toothpicks = 12)
  (h6 : lattice.boundary_toothpicks + lattice.internal_grid_toothpicks + lattice.diagonal_toothpicks = lattice.total_toothpicks) :
  ∃ (n : Nat), n = lattice.boundary_toothpicks + lattice.internal_grid_toothpicks ∧ 
               n = 16 ∧
               (∀ m : Nat, m < n → ∃ (square : Bool) (triangle : Bool), square ∨ triangle) :=
by sorry


end min_toothpicks_to_remove_l3745_374566


namespace min_colors_correct_min_colors_is_minimum_l3745_374563

-- Define a function that returns the minimum number of colors needed
def min_colors (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Edge case: no keys
  | 1 => 1
  | 2 => 2
  | _ => 3

-- Theorem statement
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n = 0 then 0
    else if n = 1 then 1
    else if n = 2 then 2
    else 3 :=
by sorry

-- Theorem stating that this is indeed the minimum
theorem min_colors_is_minimum (n : ℕ) :
  ∀ (m : ℕ), m < min_colors n → ¬(∃ (coloring : Fin n → Fin m), ∀ (i j : Fin n), i ≠ j → coloring i ≠ coloring j) :=
by sorry

end min_colors_correct_min_colors_is_minimum_l3745_374563


namespace quadratic_two_zeros_a_range_l3745_374525

theorem quadratic_two_zeros_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 0 :=
by sorry

end quadratic_two_zeros_a_range_l3745_374525


namespace olympiad_1958_l3745_374526

theorem olympiad_1958 (n : ℤ) : 1155^1958 + 34^1958 ≠ n^2 := by
  sorry

end olympiad_1958_l3745_374526


namespace triangle_height_l3745_374528

/-- Given a triangle with angles α, β, γ and side c, mc is the height corresponding to side c -/
theorem triangle_height (α β γ c mc : ℝ) (h_angles : α + β + γ = Real.pi) 
  (h_positive : 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ) :
  mc = (c * Real.sin α * Real.sin β) / Real.sin γ :=
sorry


end triangle_height_l3745_374528


namespace no_consec_nat_prod_equals_consec_even_prod_l3745_374538

theorem no_consec_nat_prod_equals_consec_even_prod : 
  ¬∃ (m n : ℕ), m * (m + 1) = 4 * n * (n + 1) := by
sorry

end no_consec_nat_prod_equals_consec_even_prod_l3745_374538


namespace choose_4_from_10_l3745_374591

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end choose_4_from_10_l3745_374591


namespace product_equals_root_fraction_l3745_374544

theorem product_equals_root_fraction (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3 / 2 : ℝ) := by
  sorry

end product_equals_root_fraction_l3745_374544


namespace unique_number_with_digit_sum_14_l3745_374575

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem unique_number_with_digit_sum_14 :
  ∃! n : ℕ,
    n > 0 ∧
    n < 1000 ∧
    (toOctal n).length = 3 ∧
    sumDigits n = 14 ∧
    sumList (toOctal n) = 14 ∧
    n = 455 := by
  sorry

end unique_number_with_digit_sum_14_l3745_374575


namespace equation_solution_l3745_374519

theorem equation_solution (x y : ℝ) : 
  y = 3 * x + 1 →
  4 * y^2 + 2 * y + 5 = 3 * (8 * x^2 + 2 * y + 3) →
  x = (-3 + Real.sqrt 21) / 6 ∨ x = (-3 - Real.sqrt 21) / 6 :=
by sorry

end equation_solution_l3745_374519


namespace quadratic_minimum_l3745_374521

theorem quadratic_minimum (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 18 * x + 7
  ∀ y : ℝ, f x ≤ f y ↔ x = 3 := by
  sorry

end quadratic_minimum_l3745_374521


namespace otimes_calculation_l3745_374508

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b^2

-- State the theorem
theorem otimes_calculation :
  let x := otimes (otimes 2 4) (otimes 1 3)
  let y := otimes 2 (otimes 4 3)
  x - y = 1215 / 512 := by sorry

end otimes_calculation_l3745_374508


namespace circle_radius_doubled_l3745_374502

theorem circle_radius_doubled (r n : ℝ) : 
  (2 * π * (r + n) = 2 * (2 * π * r)) → r = n :=
by sorry

end circle_radius_doubled_l3745_374502


namespace g_inequality_l3745_374511

def g (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem g_inequality : g (3/2) < g 0 ∧ g 0 < g 3 := by
  sorry

end g_inequality_l3745_374511


namespace factor_expression_l3745_374595

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) := by
  sorry

end factor_expression_l3745_374595


namespace shopper_fraction_l3745_374516

theorem shopper_fraction (total_shoppers : ℕ) (checkout_shoppers : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : checkout_shoppers = 180) : 
  (total_shoppers - checkout_shoppers : ℚ) / total_shoppers = 5 / 8 := by
  sorry

end shopper_fraction_l3745_374516


namespace quadratic_increases_iff_l3745_374505

/-- The quadratic function y = 2x^2 - 4x - 1 increases for x > a iff a ≥ 1 -/
theorem quadratic_increases_iff (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > a ∧ x₂ > x₁ → (2*x₂^2 - 4*x₂ - 1) > (2*x₁^2 - 4*x₁ - 1)) ↔ 
  a ≥ 1 := by
sorry

end quadratic_increases_iff_l3745_374505


namespace min_side_length_l3745_374523

theorem min_side_length (EF HG : ℝ) (EG HF : ℝ) (h1 : EF = 7) (h2 : EG = 15) (h3 : HG = 10) (h4 : HF = 25) :
  ∀ FG : ℝ, (FG > EG - EF ∧ FG > HF - HG) → FG ≥ 15 :=
by sorry

end min_side_length_l3745_374523


namespace fourth_side_length_l3745_374568

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in the circle -/
  inscribed : True -- This is a placeholder for the actual condition

/-- Theorem stating that given the specific conditions, the fourth side has length 500 -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h_radius : q.radius = 300 * Real.sqrt 2)
  (h_side1 : q.side1 = 300)
  (h_side2 : q.side2 = 300)
  (h_side3 : q.side3 = 400) :
  q.side4 = 500 := by
  sorry


end fourth_side_length_l3745_374568


namespace reflection_sum_l3745_374534

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (8, 7), then m + b = 11 -/
theorem reflection_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x - 2) * (x - 8) + (y - 3) * (y - 7) = 0 ∧ 
    (x - 5) * (1 + m * m) = m * (y - 5)) → 
  m + b = 11 := by sorry

end reflection_sum_l3745_374534


namespace students_passed_l3745_374504

def total_students : ℕ := 450

def failed_breakup : ℕ := (5 * total_students) / 12

def remaining_after_breakup : ℕ := total_students - failed_breakup

def no_show : ℕ := (7 * remaining_after_breakup) / 15

def remaining_after_no_show : ℕ := remaining_after_breakup - no_show

def penalized : ℕ := 45

def remaining_after_penalty : ℕ := remaining_after_no_show - penalized

def bonus_but_failed : ℕ := remaining_after_penalty / 8

theorem students_passed :
  total_students - failed_breakup - no_show - penalized - bonus_but_failed = 84 := by
  sorry

end students_passed_l3745_374504


namespace parallelogram_area_l3745_374514

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l3745_374514


namespace cone_volume_from_half_sector_l3745_374553

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := π * r
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 3 * π * Real.sqrt 3 := by
sorry

end cone_volume_from_half_sector_l3745_374553


namespace no_triangle_two_right_angles_l3745_374555

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_is_180 : a + b + c = 180

-- Theorem: No triangle can have two right angles
theorem no_triangle_two_right_angles :
  ∀ t : Triangle, ¬(t.a = 90 ∧ t.b = 90 ∨ t.a = 90 ∧ t.c = 90 ∨ t.b = 90 ∧ t.c = 90) :=
by
  sorry

end no_triangle_two_right_angles_l3745_374555


namespace greatest_common_divisor_of_180_and_n_l3745_374550

/-- Given two positive integers 180 and n that share exactly five positive divisors,
    the greatest of these five common divisors is 27. -/
theorem greatest_common_divisor_of_180_and_n : 
  ∀ n : ℕ+, 
  (∃! (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n)) → 
  (∃ (s : Finset ℕ+), s.card = 5 ∧ (∀ d ∈ s, d ∣ 180 ∧ d ∣ n) ∧ 27 ∈ s ∧ ∀ x ∈ s, x ≤ 27) :=
by sorry


end greatest_common_divisor_of_180_and_n_l3745_374550


namespace midpoint_trajectory_l3745_374540

/-- The trajectory of the midpoint of a line segment PQ, where P moves on the unit circle and Q is fixed at (3,0) -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x_p y_p : ℝ, x_p^2 + y_p^2 = 1 ∧ x = (x_p + 3)/2 ∧ y = y_p/2) → 
  (2*x - 3)^2 + 4*y^2 = 1 := by
sorry

end midpoint_trajectory_l3745_374540


namespace sin_cos_difference_equality_l3745_374558

theorem sin_cos_difference_equality : 
  Real.sin (7 * π / 180) * Real.cos (37 * π / 180) - 
  Real.sin (83 * π / 180) * Real.sin (37 * π / 180) = -1/2 := by
  sorry

end sin_cos_difference_equality_l3745_374558


namespace cos_six_arccos_one_fourth_l3745_374574

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end cos_six_arccos_one_fourth_l3745_374574


namespace valid_outfit_count_l3745_374509

/-- Represents the colors available for clothing items -/
inductive Color
  | Tan
  | Black
  | Blue
  | Gray
  | Green
  | White
  | Yellow

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

def is_valid_outfit (o : Outfit) : Prop :=
  o.shirt.color ≠ o.pants.color ∧ o.hat.color ≠ o.pants.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 8

def pants_colors : List Color := [Color.Tan, Color.Black, Color.Blue, Color.Gray, Color.Green]

theorem valid_outfit_count :
  (∃ (valid_outfits : List Outfit),
    (∀ o ∈ valid_outfits, is_valid_outfit o) ∧
    valid_outfits.length = 255) :=
  sorry


end valid_outfit_count_l3745_374509


namespace zach_scored_42_points_l3745_374551

def ben_points : ℝ := 21.0
def total_points : ℝ := 63

def zach_points : ℝ := total_points - ben_points

theorem zach_scored_42_points :
  zach_points = 42 := by sorry

end zach_scored_42_points_l3745_374551


namespace rectangle_length_l3745_374585

/-- The length of a rectangle with given area and width -/
theorem rectangle_length (area width : ℝ) (h_area : area = 36.48) (h_width : width = 6.08) :
  area / width = 6 := by
  sorry

end rectangle_length_l3745_374585


namespace min_shift_for_symmetry_l3745_374577

open Real

theorem min_shift_for_symmetry (x : ℝ) :
  let f (x : ℝ) := cos (2 * x) + Real.sqrt 3 * sin (2 * x)
  ∃ m : ℝ, m > 0 ∧ 
    (∀ x, f (x + m) = f (-x + m)) ∧
    (∀ m' : ℝ, m' > 0 ∧ (∀ x, f (x + m') = f (-x + m')) → m ≤ m') ∧
    m = π / 6 :=
by sorry

end min_shift_for_symmetry_l3745_374577


namespace friends_coming_over_l3745_374564

theorem friends_coming_over (sandwiches_per_friend : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_friend = 3) 
  (h2 : total_sandwiches = 12) : 
  total_sandwiches / sandwiches_per_friend = 4 :=
by sorry

end friends_coming_over_l3745_374564


namespace total_machines_is_five_l3745_374567

/-- Represents the production scenario with new and old machines -/
structure ProductionScenario where
  totalProduction : ℕ
  newMachineProduction : ℕ
  oldMachineProduction : ℕ
  totalMachines : ℕ

/-- Represents the conditions of the production problem -/
def productionProblem : ProductionScenario → Prop
  | s => s.totalProduction = 9000 ∧
         s.oldMachineProduction = s.newMachineProduction / 2 ∧
         s.totalProduction = (s.totalMachines - 1) * s.newMachineProduction + s.oldMachineProduction

/-- Represents the scenario if the old machine is replaced -/
def replacedScenario (s : ProductionScenario) : ProductionScenario :=
  { totalProduction := s.totalProduction
  , newMachineProduction := s.newMachineProduction - 200
  , oldMachineProduction := s.newMachineProduction - 200
  , totalMachines := s.totalMachines }

/-- The main theorem stating that the total number of machines is 5 -/
theorem total_machines_is_five :
  ∃ s : ProductionScenario, productionProblem s ∧
    productionProblem (replacedScenario s) ∧
    s.totalMachines = 5 := by
  sorry


end total_machines_is_five_l3745_374567


namespace sum_of_four_integers_l3745_374570

theorem sum_of_four_integers (a b c d : ℤ) :
  (a + b + c) / 3 + d = 8 ∧
  (a + b + d) / 3 + c = 12 ∧
  (a + c + d) / 3 + b = 32 / 3 ∧
  (b + c + d) / 3 + a = 28 / 3 →
  a + b + c + d = 30 := by
  sorry

end sum_of_four_integers_l3745_374570


namespace total_boxes_is_6200_l3745_374522

/-- The number of boxes in Warehouse D -/
def warehouse_d : ℕ := 800

/-- The number of boxes in Warehouse C -/
def warehouse_c : ℕ := warehouse_d - 200

/-- The number of boxes in Warehouse B -/
def warehouse_b : ℕ := 2 * warehouse_c

/-- The number of boxes in Warehouse A -/
def warehouse_a : ℕ := 3 * warehouse_b

/-- The total number of boxes in all four warehouses -/
def total_boxes : ℕ := warehouse_a + warehouse_b + warehouse_c + warehouse_d

theorem total_boxes_is_6200 : total_boxes = 6200 := by
  sorry

end total_boxes_is_6200_l3745_374522


namespace problem_statement_l3745_374598

theorem problem_statement (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) :
  (40/100) * N = 120 := by
  sorry

end problem_statement_l3745_374598


namespace population_average_age_l3745_374548

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 7 / 5)
  (h_women_age : avg_age_women = 40)
  (h_men_age : avg_age_men = 30) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 215 / 6 :=
by sorry

end population_average_age_l3745_374548


namespace existence_of_integers_l3745_374599

theorem existence_of_integers (a₁ a₂ a₃ : ℕ) (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) :
  ∃ x₁ x₂ x₃ : ℤ,
    (abs x₁ + abs x₂ + abs x₃ > 0) ∧
    (a₁ * x₁ + a₂ * x₂ + a₃ * x₃ = 0) ∧
    (max (abs x₁) (max (abs x₂) (abs x₃)) < (2 / Real.sqrt 3) * Real.sqrt a₃ + 1) :=
sorry

end existence_of_integers_l3745_374599


namespace tips_fraction_is_one_third_l3745_374549

/-- Represents the income of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income that comes from tips -/
def tipFraction (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: If tips are 2/4 of salary, then 1/3 of income is from tips -/
theorem tips_fraction_is_one_third
  (income : WaitressIncome)
  (h : income.tips = (2 : ℚ) / 4 * income.salary) :
  tipFraction income = 1 / 3 := by
  sorry

#eval (1 : ℚ) / 3  -- To check the result

end tips_fraction_is_one_third_l3745_374549


namespace classroom_count_l3745_374529

theorem classroom_count (girls boys : ℕ) (h1 : girls * 4 = boys * 3) (h2 : boys = 28) : 
  girls + boys = 49 := by
sorry

end classroom_count_l3745_374529


namespace geometric_progression_perfect_square_sum_l3745_374596

/-- A geometric progression starting with 1 -/
def GeometricProgression (r : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => r^i)

/-- The sum of a list of natural numbers -/
def ListSum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem geometric_progression_perfect_square_sum :
  ∃ r₁ r₂ n₁ n₂ : ℕ,
    r₁ ≠ r₂ ∧
    n₁ ≥ 3 ∧
    n₂ ≥ 3 ∧
    IsPerfectSquare (ListSum (GeometricProgression r₁ n₁)) ∧
    IsPerfectSquare (ListSum (GeometricProgression r₂ n₂)) :=
by sorry

end geometric_progression_perfect_square_sum_l3745_374596


namespace fraction_product_simplification_l3745_374542

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end fraction_product_simplification_l3745_374542


namespace loop_iterations_count_l3745_374588

theorem loop_iterations_count (i : ℕ) : 
  i = 20 → (∀ n : ℕ, n < 20 → i - n > 0) ∧ (i - 20 = 0) := by sorry

end loop_iterations_count_l3745_374588


namespace inequality_solution_l3745_374578

theorem inequality_solution : 
  let S : Set ℚ := {-3, -1/2, 1/3, 2}
  ∀ x ∈ S, 2*(x-1)+3 < 0 ↔ x = -3 := by sorry

end inequality_solution_l3745_374578


namespace misread_number_correction_l3745_374580

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_number : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 19)
  (h3 : correct_avg = 24)
  (h4 : incorrect_number = 26) :
  ∃ (correct_number : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = correct_number - incorrect_number ∧
    correct_number = 76 := by
  sorry

end misread_number_correction_l3745_374580


namespace truck_wash_price_l3745_374537

/-- Proves that the price of a truck wash is $6 given the conditions of Laura's carwash --/
theorem truck_wash_price (car_price : ℕ) (suv_price : ℕ) (total_raised : ℕ) 
  (num_cars num_suvs num_trucks : ℕ) :
  car_price = 5 →
  suv_price = 7 →
  num_cars = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  total_raised = 100 →
  ∃ (truck_price : ℕ), 
    truck_price = 6 ∧ 
    car_price * num_cars + suv_price * num_suvs + truck_price * num_trucks = total_raised :=
by sorry

end truck_wash_price_l3745_374537


namespace school_boys_count_l3745_374531

theorem school_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 1443 →
  boys + (boys - diff) = total →
  diff = 141 →
  boys = 792 := by
sorry

end school_boys_count_l3745_374531


namespace intersection_complement_equality_l3745_374541

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 3} := by sorry

end intersection_complement_equality_l3745_374541


namespace tax_amount_l3745_374539

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def net_pay : ℝ := 740

theorem tax_amount : 
  gross_pay * (1 - retirement_rate) - net_pay = 100 := by
  sorry

end tax_amount_l3745_374539


namespace negative_three_inequality_l3745_374557

theorem negative_three_inequality (a b : ℝ) (h : a < b) : -3*a > -3*b := by
  sorry

end negative_three_inequality_l3745_374557


namespace midpoint_specific_segment_l3745_374581

/-- The midpoint of a line segment in polar coordinates -/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let p₁ : ℝ × ℝ := (5, π/6)
  let p₂ : ℝ × ℝ := (5, -π/6)
  let m : ℝ × ℝ := midpoint_polar p₁.1 p₁.2 p₂.1 p₂.2
  m.1 > 0 ∧ 0 ≤ m.2 ∧ m.2 < 2*π ∧ m = (5*Real.sqrt 3/2, π/6) :=
by sorry

end midpoint_specific_segment_l3745_374581


namespace complex_square_equality_l3745_374594

theorem complex_square_equality (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (a : ℂ) + (b : ℂ) * Complex.I = 4 + 3 * Complex.I ↔ 
  ((a : ℂ) + (b : ℂ) * Complex.I) ^ 2 = 7 + 24 * Complex.I := by
  sorry

end complex_square_equality_l3745_374594


namespace fraction_cube_equality_l3745_374571

theorem fraction_cube_equality : (45000 ^ 3) / (15000 ^ 3) = 27 := by sorry

end fraction_cube_equality_l3745_374571


namespace real_axis_length_is_six_l3745_374560

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The line l with equation 4x + 3y - 20 = 0 -/
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 20 = 0

/-- The line l passes through one focus of the hyperbola C -/
def passes_through_focus (C : Hyperbola) : Prop :=
  ∃ (x y : ℝ), line_l x y ∧ x^2 - C.a^2 = C.b^2

/-- The line l is parallel to one of the asymptotes of the hyperbola C -/
def parallel_to_asymptote (C : Hyperbola) : Prop :=
  C.b / C.a = 4 / 3

/-- The theorem stating that the length of the real axis of the hyperbola C is 6 -/
theorem real_axis_length_is_six (C : Hyperbola)
  (h1 : passes_through_focus C)
  (h2 : parallel_to_asymptote C) :
  2 * C.a = 6 := by sorry

end real_axis_length_is_six_l3745_374560


namespace base_conversion_2023_l3745_374587

/-- Converts a number from base 10 to base 8 --/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2023 :
  toBase8 2023 = 3747 := by sorry

end base_conversion_2023_l3745_374587


namespace hyperbola_asymptotes_l3745_374569

/-- Given a hyperbola with imaginary axis length 2 and focal distance 2√3,
    prove that the equation of its asymptotes is y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : b = 1)  -- half of the imaginary axis length
  (h2 : c = Real.sqrt 3)  -- half of the focal distance
  : ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
    (∀ (x y : ℝ), (y = k * x ∨ y = -k * x) ↔ 
      (x^2 / (c^2 - b^2) - y^2 / b^2 = 1)) := by
  sorry

end hyperbola_asymptotes_l3745_374569


namespace function_inequality_l3745_374536

/-- Given a function f: ℝ → ℝ satisfying certain conditions, 
    prove that the set of x where f(x) > 1/e^x is (ln 3, +∞) -/
theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x > -f x) 
  (h2 : f (Real.log 3) = 1/3) :
  {x : ℝ | f x > Real.exp (-x)} = Set.Ioi (Real.log 3) := by
  sorry

end function_inequality_l3745_374536


namespace linear_equation_solution_l3745_374503

theorem linear_equation_solution (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = 2 → 2 * a - b - 4 = -2 := by
  sorry

end linear_equation_solution_l3745_374503


namespace fraction_equality_l3745_374527

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 := by
  sorry

end fraction_equality_l3745_374527


namespace max_trigonometric_product_l3745_374559

theorem max_trigonometric_product (x y z : ℝ) : 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 := by
  sorry

end max_trigonometric_product_l3745_374559


namespace cubic_root_sum_cubes_l3745_374589

theorem cubic_root_sum_cubes (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (2 - 3*ω + 4*ω^2)^3 + (3 + 2*ω - ω^2)^3 = 1191 := by
  sorry

end cubic_root_sum_cubes_l3745_374589


namespace sum_fraction_equality_l3745_374530

theorem sum_fraction_equality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ k ∈ ({2, 3, 4, 5, 6} : Set ℕ), 
    (a₁ / (k^2 + 1) + a₂ / (k^2 + 2) + a₃ / (k^2 + 3) + a₄ / (k^2 + 4) + a₅ / (k^2 + 5)) = 1 / k^2) :
  a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 57 / 64 := by
  sorry

end sum_fraction_equality_l3745_374530


namespace simplify_sum_of_fractions_l3745_374533

theorem simplify_sum_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hsum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) =
  1 / (9 - 2*y*z) + 1 / (9 - 2*x*z) + 1 / (9 - 2*x*y) := by
  sorry

end simplify_sum_of_fractions_l3745_374533


namespace circles_intersect_l3745_374579

/-- Circle C₁ with equation x² + y² + 2x + 2y - 2 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- Circle C₂ with equation x² + y² - 4x - 2y + 1 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end circles_intersect_l3745_374579


namespace money_left_l3745_374547

def initial_amount : ℕ := 20
def num_items : ℕ := 4
def cost_per_item : ℕ := 2

theorem money_left : initial_amount - (num_items * cost_per_item) = 12 := by
  sorry

end money_left_l3745_374547


namespace intersection_when_a_is_5_intersection_equals_A_iff_l3745_374518

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + (5-a)*x - 5*a ≤ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 6}

-- Define the complement of B
def complement_B : Set ℝ := {x | x < -3 ∨ 6 < x}

-- Theorem 1
theorem intersection_when_a_is_5 :
  A 5 ∩ complement_B = {x | -5 ≤ x ∧ x < -3} := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ complement_B = A a ↔ a < -3 := by sorry

end intersection_when_a_is_5_intersection_equals_A_iff_l3745_374518


namespace negative_x_implies_a_greater_than_five_thirds_l3745_374586

theorem negative_x_implies_a_greater_than_five_thirds
  (x a : ℝ) -- x and a are real numbers
  (h1 : x - 5 = -3 * a) -- given equation
  (h2 : x < 0) -- x is negative
  : a > 5/3 := by
sorry

end negative_x_implies_a_greater_than_five_thirds_l3745_374586


namespace transport_speed_problem_l3745_374500

/-- Proves that given two transports traveling in opposite directions for 2.71875 hours,
    with one transport traveling at 68 mph, and ending up 348 miles apart,
    the speed of the other transport must be 60 mph. -/
theorem transport_speed_problem (speed_b : ℝ) (time : ℝ) (distance : ℝ) (speed_a : ℝ) : 
  speed_b = 68 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_a = 60 := by sorry

end transport_speed_problem_l3745_374500


namespace sum_of_z_values_l3745_374510

-- Define the function g
def g (x : ℝ) : ℝ := (4 * x)^2 - (4 * x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, g z₁ = 8 ∧ g z₂ = 8 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end sum_of_z_values_l3745_374510


namespace willie_cream_purchase_l3745_374520

/-- The amount of cream Willie needs to buy given the total required amount and the amount he already has. -/
def cream_to_buy (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that Willie needs to buy 151 lbs. of cream. -/
theorem willie_cream_purchase : cream_to_buy 300 149 = 151 := by
  sorry

end willie_cream_purchase_l3745_374520


namespace equation_equivalence_l3745_374582

theorem equation_equivalence (x : ℝ) : 
  (x^2 + x + 1) * (3*x + 4) * (-7*x + 2) * (2*x - Real.sqrt 5) * (-12*x - 16) = 0 ↔ 
  (3*x + 4 = 0 ∨ -7*x + 2 = 0 ∨ 2*x - Real.sqrt 5 = 0 ∨ -12*x - 16 = 0) :=
by sorry

end equation_equivalence_l3745_374582


namespace january_book_sales_l3745_374515

/-- Proves that the number of books sold in January is 15, given the sales in February and March,
    and the average sales across all three months. -/
theorem january_book_sales (february_sales march_sales : ℕ) (average_sales : ℚ)
  (h1 : february_sales = 16)
  (h2 : march_sales = 17)
  (h3 : average_sales = 16)
  (h4 : (january_sales + february_sales + march_sales : ℚ) / 3 = average_sales) :
  january_sales = 15 := by
  sorry

end january_book_sales_l3745_374515


namespace digit_125_of_4_div_7_l3745_374573

/-- The decimal representation of 4/7 has a 6-digit repeating sequence -/
def repeating_sequence_length : ℕ := 6

/-- The 125th digit after the decimal point in 4/7 -/
def target_digit : ℕ := 125

/-- The function that returns the nth digit in the decimal expansion of 4/7 -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_125_of_4_div_7 : nth_digit target_digit = 2 := by sorry

end digit_125_of_4_div_7_l3745_374573
