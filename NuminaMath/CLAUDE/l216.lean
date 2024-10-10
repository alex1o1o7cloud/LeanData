import Mathlib

namespace correct_average_weight_l216_21690

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 61 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.65 :=
by sorry

end correct_average_weight_l216_21690


namespace trip_time_difference_l216_21601

def speed : ℝ := 40
def distance1 : ℝ := 360
def distance2 : ℝ := 400

theorem trip_time_difference : 
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end trip_time_difference_l216_21601


namespace negation_of_existence_of_real_roots_l216_21647

theorem negation_of_existence_of_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end negation_of_existence_of_real_roots_l216_21647


namespace water_displaced_by_cube_l216_21614

/-- The volume of water displaced by a partially submerged cube in a cylinder -/
theorem water_displaced_by_cube (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h_cube_side : cube_side = 10)
  (h_cylinder_radius : cylinder_radius = 5)
  (h_cylinder_height : cylinder_height = 12) :
  ∃ (v : ℝ), v = 75 * Real.sqrt 3 ∧ v^2 = 2025 := by
  sorry

end water_displaced_by_cube_l216_21614


namespace simplify_radical_product_l216_21625

theorem simplify_radical_product : 
  (3 * 5) ^ (1/3) * (5^2 * 3^4) ^ (1/2) = 15 := by
  sorry

end simplify_radical_product_l216_21625


namespace number_divided_by_three_l216_21613

theorem number_divided_by_three : ∃ n : ℝ, n / 3 = 10 ∧ n = 30 := by
  sorry

end number_divided_by_three_l216_21613


namespace hyperbola_asymptotes_l216_21628

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 3 - x^2 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = x * Real.sqrt 3 ∨ y = -x * Real.sqrt 3

/-- Theorem: The asymptotes of the given hyperbola are y = ±√3x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y := by sorry

end hyperbola_asymptotes_l216_21628


namespace function_value_sum_l216_21668

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_value_sum (f : ℝ → ℝ) 
    (h_periodic : is_periodic f 2)
    (h_odd : is_odd f)
    (h_interval : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry


end function_value_sum_l216_21668


namespace sales_minimum_value_l216_21655

/-- A quadratic function f(x) representing monthly sales -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the minimum value of the sales function -/
theorem sales_minimum_value (p q : ℝ) 
  (h1 : f p q 1 = 10) 
  (h2 : f p q 3 = 2) : 
  ∃ x, ∀ y, f p q x ≤ f p q y ∧ f p q x = -1/4 :=
sorry

end sales_minimum_value_l216_21655


namespace complex_fraction_simplification_l216_21629

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6*I
  let z₂ : ℂ := 4 - 6*I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 := by
  sorry

end complex_fraction_simplification_l216_21629


namespace sum_of_roots_is_negative_one_l216_21641

theorem sum_of_roots_is_negative_one (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 → (∀ x, x^2 + m*x + n = 0 ↔ (x = m ∨ x = n)) → m + n = -1 := by
  sorry

end sum_of_roots_is_negative_one_l216_21641


namespace cost_difference_l216_21689

def vacation_cost (tom dorothy sammy : ℝ) : Prop :=
  tom + dorothy + sammy = 400 ∧ tom = 95 ∧ dorothy = 140 ∧ sammy = 165

theorem cost_difference (tom dorothy sammy t d : ℝ) 
  (h : vacation_cost tom dorothy sammy) :
  t - d = 45 :=
sorry

end cost_difference_l216_21689


namespace solution_sets_l216_21648

-- Define the set A as (-∞, 1)
def A : Set ℝ := Set.Iio 1

-- Define the solution set B
def B (a : ℝ) : Set ℝ :=
  if a < -1 then Set.Icc a (-1)
  else if a = -1 then {-1}
  else if -1 < a ∧ a < 0 then Set.Icc (-1) a
  else ∅

-- Theorem statement
theorem solution_sets (a : ℝ) (h1 : A = {x | a * x + (-2 * a) > 0}) :
  B a = {x | (a * x - (-2 * a)) * (x - a) ≥ 0} := by
  sorry

end solution_sets_l216_21648


namespace election_majority_l216_21611

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1300 := by
  sorry

end election_majority_l216_21611


namespace parts_production_proportion_l216_21653

/-- The relationship between parts produced per minute and total parts is direct proportion -/
theorem parts_production_proportion (parts_per_minute parts_total : ℝ → ℝ) (t : ℝ) :
  (∀ t, parts_total t = (parts_per_minute t) * t) →
  ∃ k : ℝ, ∀ t, parts_total t = k * (parts_per_minute t) := by
  sorry

end parts_production_proportion_l216_21653


namespace inequality_solution_set_l216_21673

theorem inequality_solution_set :
  ∀ x : ℝ, (1/2: ℝ)^(x - x^2) < Real.log 81 / Real.log 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end inequality_solution_set_l216_21673


namespace rectangle_perimeter_equal_triangle_area_l216_21638

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter_equal_triangle_area (a b c w : ℝ) : 
  a = 9 → b = 12 → c = 15 → w = 6 → 
  (1/2) * a * b = w * ((1/2) * a * b / w) → 
  2 * (w + ((1/2) * a * b / w)) = 30 :=
by sorry

end rectangle_perimeter_equal_triangle_area_l216_21638


namespace sunlight_is_ray_telephone_line_is_segment_l216_21621

-- Define the different types of lines
inductive LineType
  | Ray
  | LineSegment
  | StraightLine

-- Define the number of endpoints for each line type
def numberOfEndpoints (lt : LineType) : Nat :=
  match lt with
  | .Ray => 1
  | .LineSegment => 2
  | .StraightLine => 0

-- Define the light emitted by the sun
def sunlight : LineType := LineType.Ray

-- Define the line between telephone poles
def telephoneLine : LineType := LineType.LineSegment

-- Theorem stating that the light emitted by the sun is a ray
theorem sunlight_is_ray : sunlight = LineType.Ray := by sorry

-- Theorem stating that the line between telephone poles is a line segment
theorem telephone_line_is_segment : telephoneLine = LineType.LineSegment := by sorry

end sunlight_is_ray_telephone_line_is_segment_l216_21621


namespace cube_root_identity_l216_21698

theorem cube_root_identity : (2^3 * 5^6 * 7^3 : ℝ)^(1/3 : ℝ) = 350 := by sorry

end cube_root_identity_l216_21698


namespace regions_on_sphere_l216_21607

/-- 
Given n great circles on a sphere where no three circles intersect at the same point,
a_n represents the number of regions formed by these circles.
-/
def a_n (n : ℕ) : ℕ := n^2 - n + 2

/-- 
Theorem: The number of regions formed by n great circles on a sphere,
where no three circles intersect at the same point, is equal to n^2 - n + 2.
-/
theorem regions_on_sphere (n : ℕ) : 
  a_n n = n^2 - n + 2 := by sorry

end regions_on_sphere_l216_21607


namespace distributeBallsWithRedBox_eq_1808_l216_21652

/-- Number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- Number of ways to distribute 7 distinguishable balls into 3 distinguishable boxes,
    where one box (red) can contain at most 3 balls -/
def distributeBallsWithRedBox : ℕ :=
  choose 7 3 * distribute 4 2 +
  choose 7 2 * distribute 5 2 +
  choose 7 1 * distribute 6 2 +
  distribute 7 2

theorem distributeBallsWithRedBox_eq_1808 :
  distributeBallsWithRedBox = 1808 := by sorry

end distributeBallsWithRedBox_eq_1808_l216_21652


namespace sum_of_squares_zero_implies_sum_l216_21631

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0 → x + y + z = 9 := by
  sorry

end sum_of_squares_zero_implies_sum_l216_21631


namespace max_regions_lines_theorem_max_regions_circles_theorem_l216_21649

/-- The maximum number of regions in a plane divided by n lines -/
def max_regions_lines (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- The maximum number of regions in a plane divided by n circles -/
def max_regions_circles (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions in a plane divided by n lines is (n^2 + n + 2) / 2 -/
theorem max_regions_lines_theorem (n : ℕ) :
  max_regions_lines n = (n^2 + n + 2) / 2 := by sorry

/-- Theorem: The maximum number of regions in a plane divided by n circles is n^2 - n + 2 -/
theorem max_regions_circles_theorem (n : ℕ) :
  max_regions_circles n = n^2 - n + 2 := by sorry

end max_regions_lines_theorem_max_regions_circles_theorem_l216_21649


namespace workshop_workers_l216_21612

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers (avg_salary : ℕ) (tech_count : ℕ) (tech_salary : ℕ) (non_tech_salary : ℕ) :
  avg_salary = 8000 →
  tech_count = 7 →
  tech_salary = 12000 →
  non_tech_salary = 6000 →
  ∃ (total_workers : ℕ), 
    (tech_count * tech_salary + (total_workers - tech_count) * non_tech_salary) / total_workers = avg_salary ∧
    total_workers = 21 := by
  sorry

end workshop_workers_l216_21612


namespace trig_identity_simplification_l216_21616

theorem trig_identity_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end trig_identity_simplification_l216_21616


namespace all_options_incorrect_l216_21676

-- Define the types for functions
def Function := ℝ → ℝ

-- Define properties of functions
def Periodic (f : Function) : Prop := 
  ∃ T > 0, ∀ x, f (x + T) = f x

def Monotonic (f : Function) : Prop := 
  ∀ x y, x < y → f x < f y

-- Original proposition
def OriginalProposition : Prop :=
  ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Theorem to prove
theorem all_options_incorrect (original : OriginalProposition) : 
  (¬(∀ f : Function, Monotonic f → ¬(Periodic f))) ∧ 
  (¬(∀ f : Function, Periodic f → Monotonic f)) ∧ 
  (¬(∀ f : Function, Monotonic f → Periodic f)) :=
sorry

end all_options_incorrect_l216_21676


namespace usual_bus_time_l216_21695

/-- Proves that the usual time to catch the bus is 12 minutes, given that walking
    at 4/5 of the usual speed results in missing the bus by 3 minutes. -/
theorem usual_bus_time (T : ℝ) (h : (5 / 4) * T = T + 3) : T = 12 := by
  sorry

end usual_bus_time_l216_21695


namespace spider_legs_count_l216_21637

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- The total number of spider legs in the room -/
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_count : total_legs = 32 := by
  sorry

end spider_legs_count_l216_21637


namespace calculation_proof_l216_21644

theorem calculation_proof : 2014 * (1 / 19 - 1 / 53) = 68 := by
  sorry

end calculation_proof_l216_21644


namespace gcd_of_specific_numbers_l216_21681

def m : ℕ := 55555555
def n : ℕ := 5555555555

theorem gcd_of_specific_numbers : Nat.gcd m n = 5 := by
  sorry

end gcd_of_specific_numbers_l216_21681


namespace rectangle_diagonal_l216_21696

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end rectangle_diagonal_l216_21696


namespace jake_has_seven_peaches_l216_21624

-- Define the number of peaches and apples for Steven and Jake
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end jake_has_seven_peaches_l216_21624


namespace smallest_x_for_prime_abs_f_l216_21627

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def f (x : ℤ) : ℤ := 4 * x^2 - 34 * x + 21

theorem smallest_x_for_prime_abs_f :
  ∃ (x : ℤ), (∀ (y : ℤ), y < x → ¬(is_prime (Int.natAbs (f y)))) ∧
             (is_prime (Int.natAbs (f x))) ∧
             x = 1 := by
  sorry

end smallest_x_for_prime_abs_f_l216_21627


namespace geometric_sequence_iff_t_eq_neg_one_l216_21608

/-- Given a sequence {a_n} with sum of first n terms S_n = 2^n + t,
    prove it's a geometric sequence iff t = -1 -/
theorem geometric_sequence_iff_t_eq_neg_one
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h_S : ∀ n, S n = 2^n + t)
  (h_a : ∀ n, a n = S n - S (n-1)) :
  (∃ r : ℝ, ∀ n > 1, a (n+1) = r * a n) ↔ t = -1 := by
sorry

end geometric_sequence_iff_t_eq_neg_one_l216_21608


namespace theorem_1_theorem_2_l216_21643

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem 1
theorem theorem_1 (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end theorem_1_theorem_2_l216_21643


namespace brick_length_is_8_l216_21692

-- Define the surface area function for a rectangular prism
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Theorem statement
theorem brick_length_is_8 :
  ∃ (l : ℝ), l > 0 ∧ surface_area l 6 2 = 152 ∧ l = 8 :=
by sorry

end brick_length_is_8_l216_21692


namespace inverse_mod_two_million_l216_21657

/-- The multiplicative inverse of (222222 * 142857) modulo 2,000,000 is 126. -/
theorem inverse_mod_two_million : ∃ N : ℕ, 
  N < 1000000 ∧ (N * (222222 * 142857)) % 2000000 = 1 :=
by
  use 126
  sorry

end inverse_mod_two_million_l216_21657


namespace number_of_large_boats_proof_number_of_large_boats_l216_21635

theorem number_of_large_boats (total_students : ℕ) (total_boats : ℕ) 
  (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) : ℕ :=
  let number_of_large_boats := 
    total_boats - (total_students - large_boat_capacity * total_boats) / 
      (large_boat_capacity - small_boat_capacity)
  number_of_large_boats

#check number_of_large_boats 50 10 6 4 = 5

theorem proof_number_of_large_boats :
  number_of_large_boats 50 10 6 4 = 5 := by
  sorry

end number_of_large_boats_proof_number_of_large_boats_l216_21635


namespace n_squared_not_divides_factorial_l216_21680

theorem n_squared_not_divides_factorial (n : ℕ) :
  ¬(n^2 ∣ n!) ↔ n = 4 ∨ Nat.Prime n :=
sorry

end n_squared_not_divides_factorial_l216_21680


namespace quadratic_roots_l216_21665

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 0 ∧ x₂^2 + x₂ = 0) ∧ 
  x₁ = 0 ∧ x₂ = -1 :=
by sorry

end quadratic_roots_l216_21665


namespace power_product_equality_l216_21640

theorem power_product_equality (x : ℝ) : (-2 * x^2) * (-4 * x^3) = 8 * x^5 := by
  sorry

end power_product_equality_l216_21640


namespace geometric_sequence_solution_l216_21663

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_solution
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_product : a 1 * a 2 * a 3 = 216)
  (h_product_35 : a 3 * a 5 = 18)
  (h_product_48 : a 4 * a 8 = 72) :
  (∃ n : ℕ, a n = 3 * 2^(n-1) ∨ a n = 12 * (1/2)^(n-1)) ∧
  (∃ q : ℝ, q = Real.sqrt 2 ∨ q = -Real.sqrt 2) :=
sorry

end geometric_sequence_solution_l216_21663


namespace vasya_drove_two_fifths_l216_21677

/-- Represents the fraction of total distance driven by each person -/
structure DriverDistances where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- The conditions of the driving problem -/
def drivingConditions (d : DriverDistances) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

theorem vasya_drove_two_fifths :
  ∀ d : DriverDistances, drivingConditions d → d.vasya = 2 / 5 := by
  sorry

end vasya_drove_two_fifths_l216_21677


namespace partner_A_profit_share_l216_21661

/-- Calculates the share of profit for partner A in a business venture --/
theorem partner_A_profit_share 
  (initial_investment : ℕ) 
  (a_withdrawal b_withdrawal c_investment : ℕ)
  (total_profit : ℕ) :
  let a_investment_months := initial_investment * 5 + (initial_investment - a_withdrawal) * 7
  let b_investment_months := initial_investment * 5 + (initial_investment - b_withdrawal) * 7
  let c_investment_months := initial_investment * 5 + (initial_investment + c_investment) * 7
  let total_investment_months := a_investment_months + b_investment_months + c_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit = 20500 :=
by
  sorry

#check partner_A_profit_share 20000 5000 4000 6000 69900

end partner_A_profit_share_l216_21661


namespace percentage_subtraction_l216_21602

theorem percentage_subtraction (a : ℝ) (p : ℝ) (h : a - p * a = 0.94 * a) : p = 0.06 := by
  sorry

end percentage_subtraction_l216_21602


namespace string_length_problem_l216_21694

/-- The length of strings problem -/
theorem string_length_problem (red white blue : ℝ) : 
  red = 8 → 
  white = 5 * red → 
  blue = 8 * white → 
  blue = 320 := by
  sorry

end string_length_problem_l216_21694


namespace least_prime_angle_in_right_triangle_l216_21604

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem least_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
    a > b →        -- Given condition: a > b
    isPrime a →    -- a is prime
    isPrime b →    -- b is prime
    b ≥ 7 :=       -- The least possible value of b is 7
by
  sorry  -- Proof is omitted as per instructions


end least_prime_angle_in_right_triangle_l216_21604


namespace cloud_counting_proof_l216_21636

def carson_clouds : ℕ := 6

def brother_clouds : ℕ := 3 * carson_clouds

def total_clouds : ℕ := carson_clouds + brother_clouds

theorem cloud_counting_proof : total_clouds = 24 := by
  sorry

end cloud_counting_proof_l216_21636


namespace product_mod_seven_l216_21682

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end product_mod_seven_l216_21682


namespace polynomial_conclusions_l216_21651

theorem polynomial_conclusions (x a : ℝ) : 
  let M : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 2
  let N : ℝ → ℝ := λ x => x^2 - a * x + 3
  (∃! i : Fin 3, 
    (i = 0 → (M x = 0 → (13 * x) / (x^2 - 3 * x - 1) = 26 / 3)) ∧
    (i = 1 → (a = -3 → (∀ y ≥ 4, M y - N y ≥ -14) → (∃ z ≥ 4, M z - N z = -14))) ∧
    (i = 2 → (a = 0 → (M x * N x = 0 → ∃ r s : ℝ, r ≠ s ∧ M r = 0 ∧ M s = 0))))
  := by sorry

end polynomial_conclusions_l216_21651


namespace dvds_in_book_l216_21678

/-- Given a DVD book with a total capacity and some empty spaces,
    calculate the number of DVDs already in the book. -/
theorem dvds_in_book (total_capacity : ℕ) (empty_spaces : ℕ)
    (h1 : total_capacity = 126)
    (h2 : empty_spaces = 45) :
    total_capacity - empty_spaces = 81 := by
  sorry

end dvds_in_book_l216_21678


namespace binomial_expansion_coefficients_sum_l216_21656

theorem binomial_expansion_coefficients_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = 48 := by
sorry

end binomial_expansion_coefficients_sum_l216_21656


namespace equality_abs_condition_l216_21654

theorem equality_abs_condition (x y : ℝ) : 
  (x = y → abs x = abs y) ∧ 
  ∃ a b : ℝ, abs a = abs b ∧ a ≠ b := by
sorry

end equality_abs_condition_l216_21654


namespace parallel_line_slope_l216_21664

/-- A point P with coordinates depending on a parameter p -/
def P (p : ℝ) : ℝ × ℝ := (2*p, -4*p + 1)

/-- The line y = kx + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 2

/-- The theorem stating that k must be -2 for the line to be parallel to the locus of P -/
theorem parallel_line_slope (k : ℝ) : 
  (∀ p : ℝ, P p ∉ {xy : ℝ × ℝ | xy.2 = line k xy.1}) → k = -2 := by
  sorry

end parallel_line_slope_l216_21664


namespace log_216_equals_3_log_6_l216_21645

theorem log_216_equals_3_log_6 : Real.log 216 = 3 * Real.log 6 := by
  sorry

end log_216_equals_3_log_6_l216_21645


namespace least_sum_of_bases_l216_21622

theorem least_sum_of_bases (c d : ℕ+) : 
  (6 * c.val + 5 = 5 * d.val + 6) →
  (∀ c' d' : ℕ+, (6 * c'.val + 5 = 5 * d'.val + 6) → c'.val + d'.val ≥ c.val + d.val) →
  c.val + d.val = 13 := by
sorry

end least_sum_of_bases_l216_21622


namespace infinitely_many_palindromes_l216_21688

/-- A function that checks if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sequence defined in the problem -/
def x (n : ℕ) : ℕ := 2013 + 317 * n

/-- The main theorem to prove -/
theorem infinitely_many_palindromes :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ isPalindrome (x n) := by sorry

end infinitely_many_palindromes_l216_21688


namespace finite_solutions_l216_21670

/-- The function F_{n,k}(x,y) as defined in the problem -/
def F (n k x y : ℕ) : ℤ := (Nat.factorial x : ℤ) + n^k + n + 1 - y^k

/-- Theorem stating that the set of solutions is finite -/
theorem finite_solutions (n k : ℕ) (hn : n > 0) (hk : k > 1) :
  Set.Finite {p : ℕ × ℕ | F n k p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0} :=
sorry

end finite_solutions_l216_21670


namespace sin_2alpha_minus_cos_pi_minus_2alpha_l216_21603

theorem sin_2alpha_minus_cos_pi_minus_2alpha (α : Real) (h : Real.tan α = 2/3) :
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17/13 := by
  sorry

end sin_2alpha_minus_cos_pi_minus_2alpha_l216_21603


namespace consecutive_integers_around_sqrt3_l216_21632

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end consecutive_integers_around_sqrt3_l216_21632


namespace class_average_problem_l216_21620

theorem class_average_problem (x : ℝ) : 
  let total_students : ℕ := 20
  let group1_students : ℕ := 10
  let group2_students : ℕ := 10
  let group2_average : ℝ := 60
  let class_average : ℝ := 70
  (group1_students : ℝ) * x + (group2_students : ℝ) * group2_average = 
    (total_students : ℝ) * class_average → x = 80 := by
  sorry

end class_average_problem_l216_21620


namespace tshirt_original_price_l216_21687

/-- Proves that the original price of a t-shirt is $20 given the conditions of the problem -/
theorem tshirt_original_price 
  (num_friends : ℕ) 
  (discount_percent : ℚ) 
  (total_spent : ℚ) : 
  num_friends = 4 → 
  discount_percent = 1/2 → 
  total_spent = 40 → 
  (total_spent / num_friends) / (1 - discount_percent) = 20 := by
sorry

end tshirt_original_price_l216_21687


namespace tony_remaining_money_l216_21642

/-- Calculates the remaining money after expenses -/
def remaining_money (initial : ℕ) (ticket : ℕ) (hot_dog : ℕ) (soda : ℕ) : ℕ :=
  initial - ticket - hot_dog - soda

/-- Proves that Tony has $26 left after his expenses -/
theorem tony_remaining_money :
  remaining_money 50 15 5 4 = 26 := by
  sorry

#eval remaining_money 50 15 5 4

end tony_remaining_money_l216_21642


namespace talking_segment_duration_l216_21619

/-- Represents the duration of a radio show in minutes -/
def show_duration : ℕ := 3 * 60

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Represents the duration of each ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the total duration of songs played in the show in minutes -/
def song_duration : ℕ := 125

/-- Theorem stating that each talking segment lasts 10 minutes -/
theorem talking_segment_duration :
  (show_duration - num_ad_breaks * ad_break_duration - song_duration) / num_talking_segments = 10 := by
  sorry

end talking_segment_duration_l216_21619


namespace flower_shop_profit_l216_21697

-- Define the profit function
def profit (n : ℕ) : ℤ :=
  if n < 16 then 10 * n - 80 else 80

-- Define the probability distribution
def prob (x : ℤ) : ℝ :=
  if x = 60 then 0.1
  else if x = 70 then 0.2
  else if x = 80 then 0.7
  else 0

-- Define the expected value
def expected_profit : ℝ :=
  60 * prob 60 + 70 * prob 70 + 80 * prob 80

-- Define the variance
def variance_profit : ℝ :=
  (60 - expected_profit)^2 * prob 60 +
  (70 - expected_profit)^2 * prob 70 +
  (80 - expected_profit)^2 * prob 80

-- Theorem statement
theorem flower_shop_profit :
  expected_profit = 76 ∧ variance_profit = 44 :=
sorry

end flower_shop_profit_l216_21697


namespace gcd_lcm_calculation_l216_21684

theorem gcd_lcm_calculation (a b : ℕ) (ha : a = 84) (hb : b = 3780) :
  (Nat.gcd a b + Nat.lcm a b) * (Nat.lcm a b * Nat.gcd a b) - 
  (Nat.lcm a b * Nat.gcd a b) = 1227194880 := by
  sorry

end gcd_lcm_calculation_l216_21684


namespace share_distribution_l216_21615

theorem share_distribution (total : ℚ) (a b c d : ℚ) 
  (h1 : total = 1000)
  (h2 : a = b + 100)
  (h3 : a = c - 100)
  (h4 : d = b - 50)
  (h5 : d = a + 150)
  (h6 : a + b + c + d = total) :
  a = 212.5 ∧ b = 112.5 ∧ c = 312.5 ∧ d = 362.5 := by
sorry

end share_distribution_l216_21615


namespace daisy_seeds_count_l216_21633

/-- The number of daisy seeds planted by Hortense -/
def daisy_seeds : ℕ := sorry

/-- The number of sunflower seeds planted by Hortense -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def total_flowering_plants : ℕ := 28

theorem daisy_seeds_count :
  (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
   ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants ∧
  daisy_seeds = 25 :=
sorry

end daisy_seeds_count_l216_21633


namespace triangle_side_lengths_and_circumradius_l216_21669

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation,
    prove that the side lengths are 3, 4, 5 and the circumradius is 2.5 -/
theorem triangle_side_lengths_and_circumradius 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (2.5 : ℝ) = (1/2 : ℝ) * c := by
  sorry

#check triangle_side_lengths_and_circumradius

end triangle_side_lengths_and_circumradius_l216_21669


namespace necessary_but_not_sufficient_l216_21658

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end necessary_but_not_sufficient_l216_21658


namespace inscribed_circle_area_ratio_l216_21617

/-- A circle inscribed in a convex polygon -/
structure InscribedCircle where
  /-- The circumference of the inscribed circle -/
  circle_circumference : ℝ
  /-- The perimeter of the convex polygon -/
  polygon_perimeter : ℝ
  /-- The area of the inscribed circle -/
  circle_area : ℝ
  /-- The area of the convex polygon -/
  polygon_area : ℝ

/-- Theorem stating that for a circle inscribed in a convex polygon with given circumference and perimeter,
    the ratio of the circle's area to the polygon's area is 2/3 -/
theorem inscribed_circle_area_ratio
  (ic : InscribedCircle)
  (h_circle_circumference : ic.circle_circumference = 10)
  (h_polygon_perimeter : ic.polygon_perimeter = 15) :
  ic.circle_area / ic.polygon_area = 2 / 3 := by
  sorry

end inscribed_circle_area_ratio_l216_21617


namespace find_x_l216_21674

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end find_x_l216_21674


namespace complex_number_range_l216_21623

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = 2 + (a + 1) * I → Complex.abs z < 2 * Real.sqrt 2 → -3 < a ∧ a < 1 := by
  sorry

end complex_number_range_l216_21623


namespace transform_is_right_shift_graph_transform_is_right_shift_l216_21672

-- Define a continuous function f from reals to reals
variable (f : ℝ → ℝ) (hf : Continuous f)

-- Define the transformation function
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - 1)

-- Theorem stating that the transformation is equivalent to a right shift
theorem transform_is_right_shift :
  ∀ x y : ℝ, transform f x = y ↔ f (x - 1) = y :=
by sorry

-- Theorem stating that the graph of the transformed function
-- is equivalent to the original graph shifted 1 unit right
theorem graph_transform_is_right_shift :
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x ↦ (x, transform f x))) ↔
             (x - 1, y) ∈ (Set.range (λ x ↦ (x, f x))) :=
by sorry

end transform_is_right_shift_graph_transform_is_right_shift_l216_21672


namespace log_eight_three_equals_512_l216_21675

theorem log_eight_three_equals_512 (y : ℝ) :
  Real.log y / Real.log 8 = 3 → y = 512 := by
  sorry

end log_eight_three_equals_512_l216_21675


namespace negative_125_to_four_thirds_l216_21667

theorem negative_125_to_four_thirds : (-125 : ℝ) ^ (4/3) = 625 := by sorry

end negative_125_to_four_thirds_l216_21667


namespace recipe_total_cups_l216_21626

-- Define the ratio of ingredients
def butter_ratio : ℚ := 2
def flour_ratio : ℚ := 5
def sugar_ratio : ℚ := 3

-- Define the amount of sugar used
def sugar_cups : ℚ := 9

-- Theorem statement
theorem recipe_total_cups : 
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let scale_factor := sugar_cups / sugar_ratio
  let total_cups := scale_factor * total_ratio
  total_cups = 30 := by sorry

end recipe_total_cups_l216_21626


namespace sum_of_repeated_addition_and_multiplication_l216_21693

theorem sum_of_repeated_addition_and_multiplication (m n : ℕ+) :
  (m.val * 2) + (3 ^ n.val) = 2 * m.val + 3 ^ n.val := by sorry

end sum_of_repeated_addition_and_multiplication_l216_21693


namespace consecutive_integers_problem_l216_21609

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  (z = 3) →
  (5 * y = 20) := by
sorry

end consecutive_integers_problem_l216_21609


namespace price_increase_problem_l216_21646

theorem price_increase_problem (candy_initial : ℝ) (soda_initial : ℝ) 
  (candy_increase : ℝ) (soda_increase : ℝ) 
  (h1 : candy_initial = 20) 
  (h2 : soda_initial = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_initial + soda_initial = 26 := by
  sorry

#check price_increase_problem

end price_increase_problem_l216_21646


namespace cyclists_meeting_time_l216_21600

/-- Represents the time (in hours) when two cyclists A and B are 32.5 km apart -/
def time_when_apart (initial_distance : ℝ) (speed_A : ℝ) (speed_B : ℝ) (final_distance : ℝ) : Set ℝ :=
  {t : ℝ | t * (speed_A + speed_B) = initial_distance - final_distance ∨ 
           t * (speed_A + speed_B) = initial_distance + final_distance}

/-- Theorem stating that the time when cyclists A and B are 32.5 km apart is either 1 or 3 hours -/
theorem cyclists_meeting_time :
  time_when_apart 65 17.5 15 32.5 = {1, 3} := by
  sorry

end cyclists_meeting_time_l216_21600


namespace decreasing_interval_of_symmetric_quadratic_l216_21618

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem decreasing_interval_of_symmetric_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x ∈ Set.range (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  a ≠ 0 →
  ∃ (l r : ℝ), l = -2/3 ∧ r = 0 ∧
    ∀ x y, l ≤ x ∧ x < y ∧ y ≤ r → f a b y < f a b x :=
by sorry

end decreasing_interval_of_symmetric_quadratic_l216_21618


namespace min_sum_squares_l216_21659

theorem min_sum_squares (a b : ℝ) (h : (9 : ℝ) / a^2 + 4 / b^2 = 1) :
  ∃ (min : ℝ), min = 25 ∧ ∀ (x y : ℝ), (9 : ℝ) / x^2 + 4 / y^2 = 1 → x^2 + y^2 ≥ min :=
by sorry

end min_sum_squares_l216_21659


namespace dice_probability_l216_21686

def number_of_dice : ℕ := 8
def probability_even : ℚ := 1/2
def probability_odd : ℚ := 1/2

theorem dice_probability :
  (number_of_dice.choose (number_of_dice / 2)) * 
  (probability_even ^ (number_of_dice / 2)) * 
  (probability_odd ^ (number_of_dice / 2)) = 35/128 := by
  sorry

end dice_probability_l216_21686


namespace parabola_intersection_dot_product_l216_21671

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Parabola.contains (c : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * c.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def dotProduct (a b : Point) : ℝ :=
  a.x * b.x + a.y * b.y

theorem parabola_intersection_dot_product 
  (c : Parabola)
  (l : Line)
  (h1 : c.contains ⟨2, -2⟩)
  (h2 : l.m = 1 ∧ l.b = -1)
  (A B : Point)
  (h3 : c.contains A ∧ l.contains A)
  (h4 : c.contains B ∧ l.contains B)
  (h5 : A ≠ B) :
  dotProduct A B = -1 := by
  sorry

end parabola_intersection_dot_product_l216_21671


namespace sum_intersections_four_lines_l216_21683

/-- The number of intersections for a given number of lines -/
def intersections (k : ℕ) : ℕ := 
  if k ≤ 1 then 0
  else Nat.choose k 2

/-- The sum of all possible numbers of intersections for up to 4 lines -/
def sum_intersections : ℕ :=
  (List.range 5).map intersections |>.sum

/-- Theorem: The sum of all possible numbers of intersections for four distinct lines in a plane is 19 -/
theorem sum_intersections_four_lines :
  sum_intersections = 19 := by sorry

end sum_intersections_four_lines_l216_21683


namespace next_joint_performance_l216_21660

theorem next_joint_performance (ella_interval : Nat) (felix_interval : Nat) 
  (grace_interval : Nat) (hugo_interval : Nat) 
  (h1 : ella_interval = 5)
  (h2 : felix_interval = 6)
  (h3 : grace_interval = 9)
  (h4 : hugo_interval = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm ella_interval felix_interval) grace_interval) hugo_interval = 90 := by
  sorry

end next_joint_performance_l216_21660


namespace min_groups_needed_l216_21606

def total_students : ℕ := 24
def max_group_size : ℕ := 10

theorem min_groups_needed : 
  (∃ (group_size : ℕ), 
    group_size > 0 ∧ 
    group_size ≤ max_group_size ∧ 
    total_students % group_size = 0 ∧
    total_students / group_size = 3) ∧
  (∀ (n : ℕ), 
    n > 0 → 
    n < 3 → 
    (∀ (group_size : ℕ), 
      group_size > 0 → 
      group_size ≤ max_group_size → 
      total_students % group_size = 0 → 
      total_students / group_size ≠ n)) :=
by sorry

end min_groups_needed_l216_21606


namespace smallest_difference_ef_de_l216_21685

/-- Represents a triangle with integer side lengths --/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given lengths satisfy the triangle inequality --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- Theorem stating the smallest possible difference between EF and DE --/
theorem smallest_difference_ef_de (t : Triangle) : 
  t.de < t.ef ∧ t.ef ≤ t.fd ∧ 
  t.de + t.ef + t.fd = 1024 ∧
  is_valid_triangle t →
  ∀ (t' : Triangle), 
    t'.de < t'.ef ∧ t'.ef ≤ t'.fd ∧
    t'.de + t'.ef + t'.fd = 1024 ∧
    is_valid_triangle t' →
    t.ef - t.de ≤ t'.ef - t'.de ∧
    t.ef - t.de = 1 :=
by sorry

end smallest_difference_ef_de_l216_21685


namespace midSectionAreaProperty_l216_21630

-- Define a right triangular pyramid
structure RightTriangularPyramid where
  -- We don't need to define all properties, just the essential ones for our theorem
  obliqueFace : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  midSection : Set (ℝ × ℝ)   -- Representing a mid-section as a set of points in 2D

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem midSectionAreaProperty (p : RightTriangularPyramid) :
  area p.midSection = (1/4) * area p.obliqueFace := by sorry

end midSectionAreaProperty_l216_21630


namespace inequality_solution_l216_21639

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (20 - x) + Real.sqrt (20 * x - x^3) ≥ 20) : x = 20 := by
  sorry

end inequality_solution_l216_21639


namespace kevin_kangaroo_hops_l216_21666

def hop_distance (n : ℕ) (remaining : ℚ) : ℚ :=
  if n % 2 = 1 then remaining / 2 else remaining / 4

def total_distance (hops : ℕ) : ℚ :=
  let rec aux (n : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if n = 0 then acc
    else
      let dist := hop_distance n remaining
      aux (n - 1) (remaining - dist) (acc + dist)
  aux hops 2 0

theorem kevin_kangaroo_hops :
  total_distance 6 = 485 / 256 := by
  sorry

#eval total_distance 6

end kevin_kangaroo_hops_l216_21666


namespace increasing_function_condition_l216_21605

open Real

/-- The function f(x) = (ln x) / x - kx is increasing on (0, +∞) iff k ≤ -1/(2e³) -/
theorem increasing_function_condition (k : ℝ) :
  (∀ x > 0, StrictMono (λ x => (log x) / x - k * x)) ↔ k ≤ -1 / (2 * (exp 3)) :=
by sorry

end increasing_function_condition_l216_21605


namespace nuts_division_proof_l216_21610

/-- The number of boys dividing nuts -/
def num_boys : ℕ := 4

/-- The number of nuts each boy receives at the end -/
def nuts_per_boy : ℕ := 3 * num_boys

/-- The number of nuts taken by the nth boy -/
def nuts_taken (n : ℕ) : ℕ := 3 * n

/-- The remaining nuts after the nth boy's turn -/
def remaining_nuts (n : ℕ) : ℕ :=
  if n = num_boys then 0
  else 5 * (nuts_per_boy - nuts_taken n)

theorem nuts_division_proof :
  (∀ n : ℕ, n ≤ num_boys → nuts_per_boy = nuts_taken n + remaining_nuts n / 5) ∧
  remaining_nuts num_boys = 0 :=
sorry

end nuts_division_proof_l216_21610


namespace intersection_of_A_and_B_l216_21699

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l216_21699


namespace min_value_theorem_l216_21691

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 4*m + n = 1) :
  (4/m + 1/n) ≥ 25 := by
  sorry

end min_value_theorem_l216_21691


namespace largest_coin_distribution_l216_21679

theorem largest_coin_distribution (n : ℕ) : n ≤ 108 ∧ n < 120 ∧ ∃ (k : ℕ), n = 15 * k + 3 →
  ∀ m : ℕ, m < 120 ∧ ∃ (k : ℕ), m = 15 * k + 3 → m ≤ n :=
by sorry

end largest_coin_distribution_l216_21679


namespace crate_stacking_probability_l216_21650

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of stacking crates to a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of stacking 15 crates to 50ft -/
theorem crate_stacking_probability :
  let dimensions : CrateDimensions := ⟨2, 3, 5⟩
  stackProbability dimensions 15 50 = 1162161 / 14348907 := by
  sorry

end crate_stacking_probability_l216_21650


namespace equation_solution_l216_21662

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 7.61 * Real.log 3 / Real.log 2 + 2 * Real.log x / Real.log 4 = x^(Real.log 16 / Real.log 9 / (Real.log x / Real.log 3)) := by
  sorry

end equation_solution_l216_21662


namespace angle_sum_theorem_l216_21634

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the theorem
theorem angle_sum_theorem (h : A + B + C + D + E + F = 90 * n) : n = 4 := by
  sorry

end angle_sum_theorem_l216_21634
