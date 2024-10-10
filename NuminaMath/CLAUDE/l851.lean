import Mathlib

namespace marie_total_sales_l851_85142

/-- The total number of items Marie sold on Saturday -/
def total_sold (newspapers : ℝ) (magazines : ℕ) : ℝ :=
  newspapers + magazines

/-- Theorem: Marie sold 425 items in total -/
theorem marie_total_sales : total_sold 275.0 150 = 425 := by
  sorry

end marie_total_sales_l851_85142


namespace film_casting_theorem_l851_85186

theorem film_casting_theorem 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (p : ℕ) 
  (k : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_p_ge : ∀ i : Fin n, p ≥ max (a i) n) 
  (h_k_le_n : k ≤ n) : 
  ∃ (castings : Fin (p^k) → (Fin n → Fin p)),
    ∀ (roles : Fin k → Fin n) (people : Fin k → ℕ),
      (∀ i : Fin k, people i < a (roles i)) →
      (∀ i j : Fin k, i ≠ j → roles i ≠ roles j) →
      ∃ day : Fin (p^k), ∀ i : Fin k, castings day (roles i) = people i :=
sorry

end film_casting_theorem_l851_85186


namespace shared_property_of_shapes_l851_85128

-- Define the basic shape
structure Shape :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of having opposite sides parallel and equal
def has_opposite_sides_parallel_and_equal (s : Shape) : Prop :=
  let v := s.vertices
  (v 0 - v 1 = v 3 - v 2) ∧ (v 1 - v 2 = v 0 - v 3)

-- Define the specific shapes
def is_parallelogram (s : Shape) : Prop :=
  has_opposite_sides_parallel_and_equal s

def is_rectangle (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  (v 1 - v 0) • (v 2 - v 1) = 0

def is_rhombus (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  ‖v 1 - v 0‖ = ‖v 2 - v 1‖

def is_square (s : Shape) : Prop :=
  is_rectangle s ∧ is_rhombus s

-- Theorem statement
theorem shared_property_of_shapes (s : Shape) :
  (is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s) →
  has_opposite_sides_parallel_and_equal s :=
sorry

end shared_property_of_shapes_l851_85128


namespace smallest_stairs_l851_85129

theorem smallest_stairs (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end smallest_stairs_l851_85129


namespace lcm_hcf_problem_l851_85132

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 385 →
  B = 180 := by
sorry

end lcm_hcf_problem_l851_85132


namespace solve_equation_l851_85168

theorem solve_equation : ∃ r : ℝ, 5 * (r - 9) = 6 * (3 - 3 * r) + 6 ∧ r = 3 := by
  sorry

end solve_equation_l851_85168


namespace future_age_difference_l851_85155

/-- Represents the age difference between Kaylee and Matt in the future -/
def AgeDifference (x : ℕ) : Prop :=
  (8 + x) = 3 * 5

/-- Proves that the number of years into the future when Kaylee will be 3 times as old as Matt is now is 7 years -/
theorem future_age_difference : ∃ (x : ℕ), AgeDifference x ∧ x = 7 := by
  sorry

end future_age_difference_l851_85155


namespace x_value_proof_l851_85106

theorem x_value_proof (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 3 * Real.sqrt 54 ∨ x = -3 * Real.sqrt 54 := by
  sorry

end x_value_proof_l851_85106


namespace exists_odd_64digit_no_zeros_div_101_l851_85127

/-- A 64-digit natural number -/
def Digit64 : Type := { n : ℕ // n ≥ 10^63 ∧ n < 10^64 }

/-- Predicate for numbers not containing zeros -/
def NoZeros (n : ℕ) : Prop := ∀ d : ℕ, d < 64 → (n / 10^d) % 10 ≠ 0

/-- Theorem stating the existence of an odd 64-digit number without zeros that is divisible by 101 -/
theorem exists_odd_64digit_no_zeros_div_101 :
  ∃ (n : Digit64), NoZeros n.val ∧ n.val % 101 = 0 ∧ n.val % 2 = 1 := by sorry

end exists_odd_64digit_no_zeros_div_101_l851_85127


namespace largest_two_digit_divisible_by_6_ending_in_4_l851_85158

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_4 n → n ≤ 84 :=
by sorry

end largest_two_digit_divisible_by_6_ending_in_4_l851_85158


namespace projection_vector_l851_85114

/-- Given vectors a and b in ℝ², prove that the projection of a onto b is equal to the expected result. -/
theorem projection_vector (a b : ℝ × ℝ) (ha : a = (2, 4)) (hb : b = (-1, 2)) :
  let proj := (((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.1,
               ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * b.2)
  proj = (-6/5, 12/5) := by
  sorry

end projection_vector_l851_85114


namespace jamal_storage_solution_l851_85115

/-- Represents the storage problem with given file sizes and constraints -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_085 : ℕ
  files_075 : ℕ
  files_045 : ℕ
  no_mix_constraint : Bool

/-- Calculates the minimum number of disks needed for the given storage problem -/
def min_disks_needed (p : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance -/
def jamal_storage : StorageProblem :=
  { total_files := 36
  , disk_capacity := 1.44
  , files_085 := 5
  , files_075 := 15
  , files_045 := 16
  , no_mix_constraint := true }

/-- Theorem stating that the minimum number of disks needed for Jamal's storage problem is 24 -/
theorem jamal_storage_solution :
  min_disks_needed jamal_storage = 24 :=
  sorry

end jamal_storage_solution_l851_85115


namespace abc_cba_divisibility_l851_85139

theorem abc_cba_divisibility (a : ℕ) (h : a ≤ 7) :
  ∃ k : ℕ, 100 * a + 10 * (a + 1) + (a + 2) + 100 * (a + 2) + 10 * (a + 1) + a = 212 * k := by
  sorry

end abc_cba_divisibility_l851_85139


namespace total_donation_is_1684_l851_85141

/-- Represents the donations to four forest reserves --/
structure ForestDonations where
  treetown : ℝ
  forest_reserve : ℝ
  animal_preservation : ℝ
  birds_sanctuary : ℝ

/-- Theorem stating the total donation given the conditions --/
theorem total_donation_is_1684 (d : ForestDonations) : 
  d.treetown = 570 ∧ 
  d.forest_reserve = d.animal_preservation + 140 ∧
  5 * d.treetown = 4 * d.forest_reserve ∧
  5 * d.treetown = 2 * d.animal_preservation ∧
  5 * d.treetown = 3 * d.birds_sanctuary →
  d.treetown + d.forest_reserve + d.animal_preservation + d.birds_sanctuary = 1684 :=
by sorry

end total_donation_is_1684_l851_85141


namespace sum_of_interior_angles_is_180_l851_85152

-- Define a triangle in Euclidean space
structure EuclideanTriangle where
  -- We don't need to specify the exact properties of a triangle here
  -- as we're focusing on the angle sum property

-- Define the concept of interior angles of a triangle
def interior_angles (t : EuclideanTriangle) : ℝ := sorry

-- State the theorem about the sum of interior angles
theorem sum_of_interior_angles_is_180 (t : EuclideanTriangle) :
  interior_angles t = 180 := by sorry

end sum_of_interior_angles_is_180_l851_85152


namespace expression_value_l851_85150

theorem expression_value (x : ℤ) (h : x = -2) : 4 * x - 5 = -13 := by
  sorry

end expression_value_l851_85150


namespace trapezoid_is_plane_figure_l851_85145

-- Define a trapezoid
structure Trapezoid :=
  (hasParallelLines : Bool)

-- Define a plane figure
structure PlaneFigure

-- Theorem: A trapezoid is a plane figure
theorem trapezoid_is_plane_figure (t : Trapezoid) (h : t.hasParallelLines = true) : PlaneFigure :=
sorry

end trapezoid_is_plane_figure_l851_85145


namespace geometric_series_sum_l851_85140

theorem geometric_series_sum : 
  let series := [2, 6, 18, 54, 162, 486, 1458, 4374]
  series.sum = 6560 := by
sorry

end geometric_series_sum_l851_85140


namespace geometric_sequence_ratio_l851_85104

/-- Given a geometric sequence {a_n} with common ratio q,
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                  -- first given condition
  a 4 + a 6 = 5/4 →                 -- second given condition
  q = 1/2 := by
sorry

end geometric_sequence_ratio_l851_85104


namespace x_equals_four_l851_85184

theorem x_equals_four (a : ℝ) (x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : 
  x = 4 := by
sorry

end x_equals_four_l851_85184


namespace sum_equals_negative_twenty_six_thirds_l851_85170

theorem sum_equals_negative_twenty_six_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) : 
  a + b + c + d = -26/3 := by
sorry

end sum_equals_negative_twenty_six_thirds_l851_85170


namespace shenny_vacation_shirts_l851_85178

/-- The number of shirts Shenny needs to pack for her vacation -/
def shirts_to_pack (vacation_days : ℕ) (same_shirt_days : ℕ) (shirts_per_day : ℕ) : ℕ :=
  (vacation_days - same_shirt_days) * shirts_per_day + 1

/-- Proof that Shenny needs to pack 11 shirts for her vacation -/
theorem shenny_vacation_shirts :
  shirts_to_pack 7 2 2 = 11 :=
by sorry

end shenny_vacation_shirts_l851_85178


namespace complex_multiplication_l851_85175

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (1 + i) = -1 + i := by sorry

end complex_multiplication_l851_85175


namespace flea_misses_point_l851_85100

/-- The number of points on the circle -/
def n : ℕ := 300

/-- The set of all points on the circle -/
def Circle := Fin n

/-- The jumping pattern of the flea -/
def jump (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The set of points visited by the flea -/
def VisitedPoints : Set Circle :=
  {p | ∃ k : ℕ, p = ⟨jump k % n, sorry⟩}

/-- Theorem stating that there exists a point the flea never visits -/
theorem flea_misses_point : ∃ p : Circle, p ∉ VisitedPoints := by
  sorry

end flea_misses_point_l851_85100


namespace function_value_at_minus_ten_l851_85187

/-- Given a function f(x) = (x-6)/(x+2), prove that f(-10) = 2 -/
theorem function_value_at_minus_ten :
  let f : ℝ → ℝ := λ x ↦ (x - 6) / (x + 2)
  f (-10) = 2 := by sorry

end function_value_at_minus_ten_l851_85187


namespace largest_number_hcf_lcm_l851_85177

theorem largest_number_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 42) 
  (h2 : Nat.lcm a b = 42 * 10 * 20) : max a b = 840 := by
  sorry

end largest_number_hcf_lcm_l851_85177


namespace range_sum_l851_85195

noncomputable def f (x : ℝ) : ℝ := 1 + (2^(x+1))/(2^x + 1) + Real.sin x

theorem range_sum (k : ℝ) (h : k > 0) :
  ∃ (m n : ℝ), (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧
                (∀ y, y ∈ Set.Icc m n ↔ ∃ x ∈ Set.Icc (-k) k, f x = y) ∧
                m + n = 4 :=
sorry

end range_sum_l851_85195


namespace root_sum_theorem_l851_85124

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x

-- Define the theorem
theorem root_sum_theorem (h k : ℝ) 
  (h_root : p h = 1) 
  (k_root : p k = 5) : 
  h + k = 2 := by
  sorry

end root_sum_theorem_l851_85124


namespace house_rent_percentage_l851_85102

def monthly_salary : ℝ := 12500
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def savings : ℝ := 2500

theorem house_rent_percentage :
  let total_percentage : ℝ := food_percentage + entertainment_percentage + conveyance_percentage
  let spent_amount : ℝ := monthly_salary - savings
  let savings_percentage : ℝ := (savings / monthly_salary) * 100
  let remaining_percentage : ℝ := 100 - total_percentage - savings_percentage
  remaining_percentage = 20 := by sorry

end house_rent_percentage_l851_85102


namespace nellys_friends_l851_85185

def pizza_cost : ℕ := 12
def people_per_pizza : ℕ := 3
def babysitting_pay : ℕ := 4
def nights_babysitting : ℕ := 15

def total_earned : ℕ := babysitting_pay * nights_babysitting
def pizzas_bought : ℕ := total_earned / pizza_cost
def total_people_fed : ℕ := pizzas_bought * people_per_pizza

theorem nellys_friends (nelly : ℕ := 1) : 
  total_people_fed - nelly = 14 := by sorry

end nellys_friends_l851_85185


namespace mix_paint_intensity_theorem_l851_85153

/-- Calculates the intensity of a paint mixture after replacing a portion of the original paint with a different intensity paint. -/
def mixPaintIntensity (originalIntensity replacementIntensity fractionReplaced : ℚ) : ℚ :=
  (1 - fractionReplaced) * originalIntensity + fractionReplaced * replacementIntensity

/-- Theorem stating that mixing 10% intensity paint with 20% intensity paint in equal proportions results in 15% intensity. -/
theorem mix_paint_intensity_theorem :
  mixPaintIntensity (1/10) (1/5) (1/2) = (3/20) := by
  sorry

#eval mixPaintIntensity (1/10) (1/5) (1/2)

end mix_paint_intensity_theorem_l851_85153


namespace james_total_money_l851_85189

-- Define the currency types
inductive Currency
| USD
| EUR

-- Define the money type
structure Money where
  amount : ℚ
  currency : Currency

-- Define the exchange rate
def exchange_rate : ℚ := 1.20

-- Define James's wallet contents
def wallet_contents : List Money := [
  ⟨50, Currency.USD⟩,
  ⟨20, Currency.USD⟩,
  ⟨5, Currency.USD⟩
]

-- Define James's pocket contents
def pocket_contents : List Money := [
  ⟨20, Currency.USD⟩,
  ⟨10, Currency.USD⟩,
  ⟨5, Currency.EUR⟩
]

-- Define James's coin contents
def coin_contents : List Money := [
  ⟨0.25, Currency.USD⟩,
  ⟨0.25, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.10, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩,
  ⟨0.01, Currency.USD⟩
]

-- Function to convert EUR to USD
def convert_to_usd (m : Money) : Money :=
  match m.currency with
  | Currency.USD => m
  | Currency.EUR => ⟨m.amount * exchange_rate, Currency.USD⟩

-- Function to sum up all money in USD
def total_usd (money_list : List Money) : ℚ :=
  (money_list.map convert_to_usd).foldl (fun acc m => acc + m.amount) 0

-- Theorem statement
theorem james_total_money :
  total_usd (wallet_contents ++ pocket_contents ++ coin_contents) = 111.85 := by
  sorry

end james_total_money_l851_85189


namespace shopping_mall_profit_l851_85157

/-- Represents the cost and selling prices of items A and B, and the minimum number of type B items to purchase for a profit exceeding $380 -/
theorem shopping_mall_profit (cost_A cost_B sell_A sell_B : ℚ) (min_B : ℕ) : 
  cost_A = cost_B - 2 →
  80 / cost_A = 100 / cost_B →
  sell_A = 12 →
  sell_B = 15 →
  cost_A = 8 →
  cost_B = 10 →
  (∀ y : ℕ, y ≥ min_B → 
    (sell_A - cost_A) * (3 * y - 5 : ℚ) + (sell_B - cost_B) * y > 380) →
  min_B = 24 :=
by sorry

end shopping_mall_profit_l851_85157


namespace pure_imaginary_fraction_l851_85107

theorem pure_imaginary_fraction (b : ℝ) : 
  (Complex.I * (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I))).re = 0 → 
  (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I)).im ≠ 0 → 
  b = 2 := by
sorry

end pure_imaginary_fraction_l851_85107


namespace choir_members_l851_85136

theorem choir_members (n : ℕ) (h1 : 50 ≤ n) (h2 : n ≤ 200) 
  (h3 : n % 7 = 4) (h4 : n % 6 = 8) : 
  n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186 := by
sorry

end choir_members_l851_85136


namespace inequality_system_no_solution_l851_85169

theorem inequality_system_no_solution (a : ℝ) :
  (∀ x : ℝ, ¬(x > a + 2 ∧ x < 3*a - 2)) ↔ a ≤ 2 := by
  sorry

end inequality_system_no_solution_l851_85169


namespace remainder_101_pow_37_mod_100_l851_85160

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end remainder_101_pow_37_mod_100_l851_85160


namespace parallel_perpendicular_to_plane_l851_85197

/-- Two lines are parallel -/
def parallel (a b : Line3) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

/-- The theorem statement -/
theorem parallel_perpendicular_to_plane 
  (a b : Line3) (α : Plane3) 
  (h1 : parallel a b) 
  (h2 : perpendicular_to_plane a α) : 
  perpendicular_to_plane b α := by sorry

end parallel_perpendicular_to_plane_l851_85197


namespace solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l851_85164

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * |x - 1|

-- Theorem 1
theorem solutions_when_a_is_one :
  {x : ℝ | |f x| = g 1 x} = {-2, 0, 1} := by sorry

-- Theorem 2
theorem two_distinct_solutions (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ |f x| = g a x ∧ |f y| = g a y) ↔ (a = 0 ∨ a = 2) := by sorry

-- Theorem 3
theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end solutions_when_a_is_one_two_distinct_solutions_inequality_holds_for_all_x_l851_85164


namespace empty_solution_set_iff_k_ge_one_l851_85121

-- Define the function representing the left side of the inequality
def f (k x : ℝ) : ℝ := k * x^2 - 2 * |x - 1| + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x : ℝ, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k : ℝ, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end empty_solution_set_iff_k_ge_one_l851_85121


namespace max_triangles_in_7x7_grid_triangle_l851_85135

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Counts the maximum number of triangles in a grid triangle -/
def count_max_triangles (t : GridTriangle) : ℕ := sorry

/-- The main theorem stating the maximum number of triangles in a 7x7 grid triangle -/
theorem max_triangles_in_7x7_grid_triangle :
  ∀ (t : GridTriangle),
    t.leg_length = 7 →
    t.is_right_angled = true →
    count_max_triangles t = 28 := by sorry

end max_triangles_in_7x7_grid_triangle_l851_85135


namespace price_reduction_proof_l851_85190

/-- Given the initial price of a box of cereal, the number of boxes bought, and the total amount paid,
    prove that the price reduction per box is correct. -/
theorem price_reduction_proof (initial_price : ℕ) (boxes_bought : ℕ) (total_paid : ℕ) :
  initial_price = 104 →
  boxes_bought = 20 →
  total_paid = 1600 →
  initial_price - (total_paid / boxes_bought) = 24 := by
  sorry

end price_reduction_proof_l851_85190


namespace function_zero_point_implies_a_range_l851_85131

theorem function_zero_point_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 1 ∧ 4 * |a| * x₀ - 2 * a + 1 = 0) →
  a > 1/2 := by
  sorry

end function_zero_point_implies_a_range_l851_85131


namespace number_division_problem_l851_85162

theorem number_division_problem (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 - 3 = 247 := by
  sorry

end number_division_problem_l851_85162


namespace exists_polygon_with_n_triangulations_l851_85118

/-- A polygon is a closed planar figure with straight sides. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon for this statement
  mk :: (dummy : Unit)

/-- The number of triangulations of a polygon. -/
def numTriangulations (p : Polygon) : ℕ := sorry

/-- For any positive integer n, there exists a polygon with exactly n triangulations. -/
theorem exists_polygon_with_n_triangulations :
  ∀ n : ℕ, n > 0 → ∃ p : Polygon, numTriangulations p = n := by sorry

end exists_polygon_with_n_triangulations_l851_85118


namespace yans_distance_ratio_l851_85103

theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
    w > 0 →  -- walking speed is positive
    x > 0 →  -- distance from Yan to home is positive
    y > 0 →  -- distance from Yan to stadium is positive
    y / w = x / w + (x + y) / (10 * w) →  -- time equality condition
    x / y = 9 / 11 := by
  sorry

end yans_distance_ratio_l851_85103


namespace vitamin_c_in_two_juices_l851_85194

/-- Amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℝ := 103

/-- Amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℝ := 82

/-- Total amount of vitamin C (in mg) in two glasses of apple juice and three glasses of orange juice -/
def total_vc_five_glasses : ℝ := 452

/-- Theorem stating that one glass each of apple and orange juice contain 185 mg of vitamin C -/
theorem vitamin_c_in_two_juices :
  apple_juice_vc + orange_juice_vc = 185 ∧
  2 * apple_juice_vc + 3 * orange_juice_vc = total_vc_five_glasses :=
sorry

end vitamin_c_in_two_juices_l851_85194


namespace stating_whack_a_mole_tickets_correct_l851_85137

/-- Represents the number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- Represents the number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- Represents the number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Represents the number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- 
Theorem stating that the number of tickets Tom won from 'whack a mole' 
is correct given the other known information
-/
theorem whack_a_mole_tickets_correct : 
  whack_a_mole_tickets + skee_ball_tickets = remaining_tickets + spent_tickets := by
  sorry

#check whack_a_mole_tickets_correct

end stating_whack_a_mole_tickets_correct_l851_85137


namespace triangle_perimeter_l851_85156

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 6√2 under the given conditions. -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) : 
  A = π / 3 →
  b + c = 2 * a →
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3 →
  a + b + c = 6 * Real.sqrt 2 := by
  sorry

end triangle_perimeter_l851_85156


namespace fifth_number_ninth_row_is_61_l851_85188

/-- Represents a lattice pattern with a given number of columns per row -/
structure LatticePattern where
  columnsPerRow : ℕ

/-- Calculates the last number in a given row of the lattice pattern -/
def lastNumberInRow (pattern : LatticePattern) (row : ℕ) : ℕ :=
  pattern.columnsPerRow * row

/-- Calculates the nth number from the end in a given row -/
def nthNumberFromEnd (pattern : LatticePattern) (row : ℕ) (n : ℕ) : ℕ :=
  lastNumberInRow pattern row - (n - 1)

/-- The theorem to be proved -/
theorem fifth_number_ninth_row_is_61 :
  let pattern : LatticePattern := ⟨7⟩
  nthNumberFromEnd pattern 9 5 = 61 := by
  sorry

end fifth_number_ninth_row_is_61_l851_85188


namespace sock_purchase_theorem_l851_85166

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.four_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 4 * p.four_dollar + 5 * p.five_dollar = 38 ∧
  p.two_dollar ≥ 1 ∧ p.four_dollar ≥ 1 ∧ p.five_dollar ≥ 1

theorem sock_purchase_theorem :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 12 :=
by sorry

end sock_purchase_theorem_l851_85166


namespace isosceles_triangle_largest_angle_l851_85182

theorem isosceles_triangle_largest_angle (a b c : ℝ) : 
  -- The triangle is isosceles
  a = b →
  -- One of the angles opposite an equal side is 50°
  c = 50 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 :=
by sorry

end isosceles_triangle_largest_angle_l851_85182


namespace particle_speed_l851_85109

/-- A particle moves in a 2D plane. Its position at time t is given by (3t + 1, -2t + 5). 
    The theorem states that the speed of the particle is √13 units of distance per unit of time. -/
theorem particle_speed (t : ℝ) : 
  let position := fun (t : ℝ) => (3 * t + 1, -2 * t + 5)
  let velocity := fun (t : ℝ) => (3, -2)
  let speed := Real.sqrt (3^2 + (-2)^2)
  speed = Real.sqrt 13 := by sorry

end particle_speed_l851_85109


namespace root_sum_product_l851_85198

theorem root_sum_product (a b : ℝ) : 
  (a^4 + 2*a^3 - 4*a - 1 = 0) →
  (b^4 + 2*b^3 - 4*b - 1 = 0) →
  (a ≠ b) →
  (a*b + a + b = Real.sqrt 3 - 2) := by
sorry

end root_sum_product_l851_85198


namespace rower_round_trip_time_l851_85123

/-- Proves that the total time to row to Big Rock and back is 1 hour -/
theorem rower_round_trip_time
  (rower_speed : ℝ)
  (river_speed : ℝ)
  (distance : ℝ)
  (h1 : rower_speed = 7)
  (h2 : river_speed = 2)
  (h3 : distance = 3.2142857142857144)
  : (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1 := by
  sorry


end rower_round_trip_time_l851_85123


namespace g_neg_four_l851_85120

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_neg_four : g (-4) = -18 := by
  sorry

end g_neg_four_l851_85120


namespace max_value_a_l851_85112

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
sorry

end max_value_a_l851_85112


namespace solution_to_equation_l851_85171

theorem solution_to_equation : ∃ x : ℝ, 12*x + 13*x + 16*x + 11 = 134 ∧ x = 3 := by
  sorry

end solution_to_equation_l851_85171


namespace cover_cost_is_77_l851_85122

/-- Represents the cost of printing a book in kopecks -/
def book_cost (cover_cost : ℕ) (page_cost : ℕ) (num_pages : ℕ) : ℕ :=
  (cover_cost * 100 + page_cost * num_pages + 99) / 100 * 100

/-- The problem statement -/
theorem cover_cost_is_77 : 
  ∃ (cover_cost page_cost : ℕ),
    (∀ n, book_cost cover_cost page_cost n = ((cover_cost * 100 + page_cost * n + 99) / 100) * 100) ∧
    book_cost cover_cost page_cost 104 = 134 * 100 ∧
    book_cost cover_cost page_cost 192 = 181 * 100 ∧
    cover_cost = 77 :=
by
  sorry

end cover_cost_is_77_l851_85122


namespace tangent_parallel_condition_extreme_values_and_intersections_l851_85130

/-- The cubic function f(x) = x^3 - ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem tangent_parallel_condition (a b c : ℝ) :
  (∃ x₀ : ℝ, f' a b x₀ = 0) → a^2 ≥ 3*b :=
sorry

theorem extreme_values_and_intersections (c : ℝ) :
  (f' 3 (-9) (-1) = 0 ∧ f' 3 (-9) 3 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f 3 (-9) c x₁ = 0 ∧ f 3 (-9) c x₂ = 0 ∧ f 3 (-9) c x₃ = 0 ∧
    (∀ x : ℝ, f 3 (-9) c x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  -5 < c ∧ c < 27 :=
sorry

end tangent_parallel_condition_extreme_values_and_intersections_l851_85130


namespace f_properties_l851_85101

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x + g' x - 10 = 0
axiom cond2 : ∀ x, f x - g' (4 - x) - 10 = 0
axiom g_even : ∀ x, g (-x) = g x

-- State the theorem
theorem f_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end f_properties_l851_85101


namespace factorization_equality_l851_85116

theorem factorization_equality (a b : ℝ) : (a - b)^2 - (b - a) = (a - b) * ((a - b) + 1) := by
  sorry

end factorization_equality_l851_85116


namespace equal_sums_exist_l851_85191

/-- A 3x3 table with entries of 1, 0, or -1 -/
def Table := Fin 3 → Fin 3 → Int

/-- Predicate to check if a table is valid (contains only 1, 0, or -1) -/
def isValidTable (t : Table) : Prop :=
  ∀ i j, t i j = 1 ∨ t i j = 0 ∨ t i j = -1

/-- Sum of a row in the table -/
def rowSum (t : Table) (i : Fin 3) : Int :=
  (t i 0) + (t i 1) + (t i 2)

/-- Sum of a column in the table -/
def colSum (t : Table) (j : Fin 3) : Int :=
  (t 0 j) + (t 1 j) + (t 2 j)

/-- List of all row and column sums -/
def allSums (t : Table) : List Int :=
  (List.range 3).map (rowSum t) ++ (List.range 3).map (colSum t)

/-- Theorem: In a valid 3x3 table, there exist at least two equal sums among row and column sums -/
theorem equal_sums_exist (t : Table) (h : isValidTable t) :
  ∃ (x y : Fin 6), x ≠ y ∧ (allSums t).get x = (allSums t).get y :=
sorry

end equal_sums_exist_l851_85191


namespace player_A_can_win_l851_85149

/-- Represents a game board with three rows --/
structure GameBoard :=
  (row1 : List ℤ)
  (row2 : List ℤ)
  (row3 : List ℤ)

/-- Represents a player in the game --/
inductive Player
  | A
  | B

/-- Defines a valid game board configuration --/
def ValidBoard (board : GameBoard) : Prop :=
  Odd board.row1.length ∧ 
  Odd board.row2.length ∧ 
  Odd board.row3.length

/-- Defines the game state --/
structure GameState :=
  (board : GameBoard)
  (currentPlayer : Player)

/-- Defines a game strategy for player A --/
def Strategy := GameState → ℕ → ℤ → GameState

/-- Theorem: Player A can always achieve the desired row sums --/
theorem player_A_can_win (initialBoard : GameBoard) (targetSum1 targetSum2 targetSum3 : ℤ) :
  ValidBoard initialBoard →
  ∃ (strategy : Strategy),
    (∀ (finalBoard : GameBoard),
      (finalBoard.row1.sum = targetSum1) ∧
      (finalBoard.row2.sum = targetSum2) ∧
      (finalBoard.row3.sum = targetSum3)) :=
sorry

end player_A_can_win_l851_85149


namespace t_shaped_area_l851_85154

/-- The area of a T-shaped region formed by subtracting three smaller rectangles
    from a larger rectangle -/
theorem t_shaped_area (total_width total_height : ℝ)
                      (rect1_width rect1_height : ℝ)
                      (rect2_width rect2_height : ℝ)
                      (rect3_width rect3_height : ℝ)
                      (h1 : total_width = 6)
                      (h2 : total_height = 5)
                      (h3 : rect1_width = 1)
                      (h4 : rect1_height = 4)
                      (h5 : rect2_width = 1)
                      (h6 : rect2_height = 4)
                      (h7 : rect3_width = 1)
                      (h8 : rect3_height = 3) :
  total_width * total_height - 
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) = 19 := by
  sorry

end t_shaped_area_l851_85154


namespace blueberries_needed_for_pies_l851_85148

-- Define the constants
def blueberries_per_pint : ℕ := 200
def pints_per_quart : ℕ := 2
def pies_to_make : ℕ := 6

-- Define the theorem
theorem blueberries_needed_for_pies : 
  blueberries_per_pint * pints_per_quart * pies_to_make = 2400 := by
  sorry

end blueberries_needed_for_pies_l851_85148


namespace money_distribution_l851_85172

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 340) :
  C = 40 := by sorry

end money_distribution_l851_85172


namespace fraction_sum_mixed_number_equality_main_theorem_l851_85196

theorem fraction_sum : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (4 : ℚ) / 3 = (35 : ℚ) / 12 := by
  sorry

theorem mixed_number_equality : (35 : ℚ) / 12 = 2 + (11 : ℚ) / 12 := by
  sorry

theorem main_theorem : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (1 + (1 : ℚ) / 3) = 2 + (11 : ℚ) / 12 := by
  sorry

end fraction_sum_mixed_number_equality_main_theorem_l851_85196


namespace complex_equation_solution_l851_85183

theorem complex_equation_solution :
  ∀ z : ℂ, (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = (-1/5 : ℂ) - (3/5 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l851_85183


namespace prime_divisors_inequality_l851_85108

-- Define the variables
variable (x y z : ℕ)
variable (p q : ℕ)

-- Define the conditions
variable (h1 : x > 2)
variable (h2 : y > 1)
variable (h3 : z > 0)
variable (h4 : x^y + 1 = z^2)

-- Define p and q
variable (hp : p = (Nat.factors x).card)
variable (hq : q = (Nat.factors y).card)

-- State the theorem
theorem prime_divisors_inequality : p ≥ q + 2 := by
  sorry

end prime_divisors_inequality_l851_85108


namespace inches_per_foot_l851_85143

theorem inches_per_foot (rope_first : ℕ) (rope_difference : ℕ) (total_inches : ℕ) : 
  rope_first = 6 →
  rope_difference = 4 →
  total_inches = 96 →
  (total_inches / (rope_first + (rope_first - rope_difference))) = 12 :=
by sorry

end inches_per_foot_l851_85143


namespace special_sum_eq_1010_l851_85110

/-- Double factorial of a natural number -/
def doubleFac : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * doubleFac n

/-- The sum from i=1 to 1010 of ((2i)!! / (2i+1)!!) * ((2i+1)! / (2i)!) -/
def specialSum : ℚ :=
  (Finset.range 1010).sum (fun i =>
    let i' := i + 1
    (doubleFac (2 * i') : ℚ) / (doubleFac (2 * i' + 1)) *
    (Nat.factorial (2 * i' + 1) : ℚ) / (Nat.factorial (2 * i')))

/-- The sum is equal to 1010 -/
theorem special_sum_eq_1010 : specialSum = 1010 := by sorry

end special_sum_eq_1010_l851_85110


namespace max_distance_to_complex_point_l851_85181

open Complex

theorem max_distance_to_complex_point (z : ℂ) :
  let z₁ : ℂ := 2 - 2*I
  (abs z = 1) →
  (∀ w : ℂ, abs w = 1 → abs (w - z₁) ≤ 2*Real.sqrt 2 + 1) ∧
  (∃ w : ℂ, abs w = 1 ∧ abs (w - z₁) = 2*Real.sqrt 2 + 1) :=
by sorry

end max_distance_to_complex_point_l851_85181


namespace shopping_remaining_amount_l851_85173

theorem shopping_remaining_amount (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : spent_percentage = 0.30) : 
  initial_amount - (spent_percentage * initial_amount) = 3500 := by
  sorry

end shopping_remaining_amount_l851_85173


namespace height_on_longest_side_of_6_8_10_triangle_l851_85165

theorem height_on_longest_side_of_6_8_10_triangle :
  ∃ (a b c h : ℝ),
    a = 6 ∧ b = 8 ∧ c = 10 ∧
    a^2 + b^2 = c^2 ∧
    c > a ∧ c > b ∧
    h = 4.8 ∧
    (1/2) * c * h = (1/2) * a * b :=
sorry

end height_on_longest_side_of_6_8_10_triangle_l851_85165


namespace square_difference_division_l851_85174

theorem square_difference_division (a b c : ℕ) (h : (a^2 - b^2) / c = 488) :
  (144^2 - 100^2) / 22 = 488 := by
  sorry

end square_difference_division_l851_85174


namespace unique_solution_l851_85105

/-- Represents the ages of two sons given specific conditions --/
structure SonsAges where
  elder : ℕ
  younger : ℕ
  doubled_elder_exceeds_sum : 2 * elder = elder + younger + 18
  younger_less_than_difference : younger = elder - younger - 6

/-- The unique solution to the SonsAges problem --/
def solution : SonsAges := { 
  elder := 30,
  younger := 12,
  doubled_elder_exceeds_sum := by sorry,
  younger_less_than_difference := by sorry
}

/-- Proves that the solution is unique --/
theorem unique_solution (s : SonsAges) : s = solution := by sorry

end unique_solution_l851_85105


namespace det_circulant_matrix_l851_85111

def circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => if i = j then 2
           else if (i - j) % n = 2 ∨ (i - j) % n = n - 2 then 1
           else 0

theorem det_circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) :
  let A := circulant_matrix n h h_odd
  Matrix.det A = 4 := by
  sorry

end det_circulant_matrix_l851_85111


namespace team_leader_deputy_count_l851_85146

def people : Nat := 5

theorem team_leader_deputy_count : 
  (people * (people - 1) : Nat) = 20 := by
  sorry

end team_leader_deputy_count_l851_85146


namespace election_result_l851_85163

/-- Represents the total number of votes in the election --/
def total_votes : ℕ := sorry

/-- Represents the initial percentage of votes for Candidate A --/
def initial_votes_A : ℚ := 65 / 100

/-- Represents the initial percentage of votes for Candidate B --/
def initial_votes_B : ℚ := 50 / 100

/-- Represents the initial percentage of votes for Candidate C --/
def initial_votes_C : ℚ := 45 / 100

/-- Represents the number of votes that change from A to B --/
def votes_A_to_B : ℕ := 1000

/-- Represents the number of votes that change from C to B --/
def votes_C_to_B : ℕ := 500

/-- Represents the final percentage of votes for Candidate B --/
def final_votes_B : ℚ := 70 / 100

theorem election_result : 
  initial_votes_B * total_votes + votes_A_to_B + votes_C_to_B = final_votes_B * total_votes ∧
  total_votes = 7500 := by sorry

end election_result_l851_85163


namespace unique_b_values_l851_85180

theorem unique_b_values : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 15 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 3 := by
  sorry

end unique_b_values_l851_85180


namespace distance_of_problem_lines_l851_85144

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on the first line
  b : ℝ × ℝ  -- Point on the second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { a := (3, -2)
    b := (5, -1)
    d := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = 2 * Real.sqrt 109 / 29 :=
sorry

end distance_of_problem_lines_l851_85144


namespace function_growth_l851_85179

open Real

theorem function_growth (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_growth : ∀ x, (deriv f) x > f x ∧ f x > 0) : 
  f 8 > 2022 * f 0 := by
  sorry

end function_growth_l851_85179


namespace equation1_no_solution_equation2_unique_solution_l851_85113

-- Define the equations
def equation1 (x : ℝ) : Prop := (4 - x) / (x - 3) + 1 / (3 - x) = 1
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_no_solution : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem equation2_unique_solution : ∃! x : ℝ, equation2 x ∧ x = 2 := by sorry

end equation1_no_solution_equation2_unique_solution_l851_85113


namespace paths_from_A_to_D_l851_85192

/-- Represents a point in the graph -/
inductive Point
| A
| B
| C
| D

/-- Represents the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  match start, finish with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ :=
  (num_paths Point.A Point.B) * (num_paths Point.B Point.C) * (num_paths Point.C Point.D) +
  (num_paths Point.A Point.C) * (num_paths Point.C Point.D)

theorem paths_from_A_to_D : total_paths = 10 := by
  sorry

end paths_from_A_to_D_l851_85192


namespace part_one_part_two_l851_85159

-- Definitions for propositions q and p
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0

def prop_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 4 < 0

-- Part 1
theorem part_one (a : ℝ) : prop_q a ∨ prop_p a → a < -3 ∨ a ≥ -1 := by sorry

-- Definitions for part 2
def prop_p_part2 (a : ℝ) : Prop := ∃ x : ℝ, 2*a < x ∧ x < a + 1

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, prop_p_part2 a → prop_q a) ∧ 
  (∃ x : ℝ, prop_q a ∧ ¬prop_p_part2 a) → 
  a ≥ -1/2 := by sorry

end part_one_part_two_l851_85159


namespace min_distance_complex_l851_85125

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_l851_85125


namespace unknown_number_problem_l851_85199

theorem unknown_number_problem (x : ℝ) : 
  (0.1 * 30 + 0.15 * x = 10.5) → x = 50 := by
  sorry

end unknown_number_problem_l851_85199


namespace product_sum_bounds_l851_85151

def pairProductSum (pairs : List (ℕ × ℕ)) : ℕ :=
  (pairs.map (λ (a, b) => a * b)).sum

theorem product_sum_bounds :
  ∀ (pairs : List (ℕ × ℕ)),
    pairs.length = 50 ∧
    (pairs.map Prod.fst ++ pairs.map Prod.snd).toFinset = Finset.range 100
    →
    85850 ≤ pairProductSum pairs ∧ pairProductSum pairs ≤ 169150 :=
by sorry

end product_sum_bounds_l851_85151


namespace vector_perpendicular_l851_85147

theorem vector_perpendicular (a b : ℝ × ℝ) :
  a = (2, 0) →
  b = (-1, 1) →
  b • (a + b) = 0 := by
sorry

end vector_perpendicular_l851_85147


namespace ordering_of_a_b_c_l851_85138

theorem ordering_of_a_b_c :
  let a := Real.tan (1/2)
  let b := Real.tan (2/π)
  let c := Real.sqrt 3 / π
  a < c ∧ c < b := by sorry

end ordering_of_a_b_c_l851_85138


namespace sequence_increasing_l851_85119

theorem sequence_increasing (n : ℕ) (h : n ≥ 1) : 
  let a : ℕ → ℚ := fun k => (2 * k : ℚ) / (3 * k + 1)
  a (n + 1) > a n := by
sorry

end sequence_increasing_l851_85119


namespace least_n_for_inequality_l851_85133

theorem least_n_for_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k > 0 → (1 / k - 1 / (k + 1) < 1 / 15) → k ≥ n) ∧ (1 / n - 1 / (n + 1) < 1 / 15) ∧ n = 4 := by
  sorry

end least_n_for_inequality_l851_85133


namespace spinner_probabilities_l851_85134

theorem spinner_probabilities :
  ∀ (p_C : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + p_C + p_C = 1 →
  p_C = (5 / 24 : ℚ) :=
by
  sorry

end spinner_probabilities_l851_85134


namespace venus_speed_mph_l851_85167

/-- The speed of Venus in miles per second -/
def venus_speed_mps : ℝ := 21.9

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: Venus's speed in miles per hour -/
theorem venus_speed_mph : ⌊venus_speed_mps * seconds_per_hour⌋ = 78840 := by
  sorry

end venus_speed_mph_l851_85167


namespace convex_polygon_angle_sum_l851_85117

theorem convex_polygon_angle_sum (n : ℕ) (angle_sum : ℝ) : n = 17 → angle_sum = 2610 → ∃ (missing_angle : ℝ), 
  0 < missing_angle ∧ 
  missing_angle < 180 ∧ 
  (180 * (n - 2) : ℝ) = angle_sum + missing_angle := by
  sorry

end convex_polygon_angle_sum_l851_85117


namespace marbles_selection_count_l851_85126

/-- The number of ways to choose 4 marbles from 8, with at least one red -/
def choose_marbles (total_marbles : ℕ) (red_marbles : ℕ) (choose : ℕ) : ℕ :=
  Nat.choose (total_marbles - red_marbles) (choose - 1)

/-- Theorem: There are 35 ways to choose 4 marbles from 8, with at least one red -/
theorem marbles_selection_count :
  choose_marbles 8 1 4 = 35 := by
  sorry

end marbles_selection_count_l851_85126


namespace root_transformation_l851_85193

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3 * r₁^2 + 13 = 0) ∧ 
  (r₂^3 - 3 * r₂^2 + 13 = 0) ∧ 
  (r₃^3 - 3 * r₃^2 + 13 = 0) →
  ((3 * r₁)^3 - 9 * (3 * r₁)^2 + 351 = 0) ∧
  ((3 * r₂)^3 - 9 * (3 * r₂)^2 + 351 = 0) ∧
  ((3 * r₃)^3 - 9 * (3 * r₃)^2 + 351 = 0) :=
by sorry

end root_transformation_l851_85193


namespace smallest_composite_no_small_primes_l851_85161

/-- A function that returns true if a number is composite, false otherwise -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that returns true if a number has no prime factors less than 20, false otherwise -/
def no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_primes : 
  (is_composite 529 ∧ no_small_prime_factors 529) ∧ 
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_primes_l851_85161


namespace total_students_l851_85176

theorem total_students (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 →
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 :=
by
  sorry

end total_students_l851_85176
