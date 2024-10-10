import Mathlib

namespace margo_walking_distance_l2318_231899

/-- Proves that Margo's total walking distance is 2 miles given the specified conditions -/
theorem margo_walking_distance
  (time_to_friend : ℝ)
  (time_to_return : ℝ)
  (average_speed : ℝ)
  (h1 : time_to_friend = 15)
  (h2 : time_to_return = 25)
  (h3 : average_speed = 3)
  : (time_to_friend + time_to_return) / 60 * average_speed = 2 := by
  sorry

end margo_walking_distance_l2318_231899


namespace train_length_l2318_231885

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 127 → time = 17 → ∃ (length : ℝ), abs (length - 599.76) < 0.01 := by
  sorry

end train_length_l2318_231885


namespace total_laundry_time_l2318_231883

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTimeForLoad (load : LaundryTime) : ℕ :=
  load.washing + load.drying

/-- The time for the whites load -/
def whites : LaundryTime := ⟨72, 50⟩

/-- The time for the darks load -/
def darks : LaundryTime := ⟨58, 65⟩

/-- The time for the colors load -/
def colors : LaundryTime := ⟨45, 54⟩

/-- Theorem stating that the total time for all three loads is 344 minutes -/
theorem total_laundry_time :
  totalTimeForLoad whites + totalTimeForLoad darks + totalTimeForLoad colors = 344 := by
  sorry

end total_laundry_time_l2318_231883


namespace triangle_angle_problem_l2318_231812

theorem triangle_angle_problem (A B C : ℝ) : 
  A = 32 ∧ 
  C = 2 * A - 12 ∧ 
  B = 3 * A ∧ 
  A + B + C = 180 → 
  B = 96 := by
sorry

end triangle_angle_problem_l2318_231812


namespace camp_attendance_outside_county_attendance_l2318_231869

theorem camp_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  lawrence_camp = lawrence_total - lawrence_home :=
by sorry

theorem outside_county_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  0 = lawrence_camp - (lawrence_total - lawrence_home) :=
by sorry

end camp_attendance_outside_county_attendance_l2318_231869


namespace train_length_calculation_l2318_231866

/-- Calculates the length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  let train_speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := train_speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ bridge_length = 219.03 ∧ crossing_time = 30 →
  train_length = 155.97 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2318_231866


namespace intersection_of_M_and_N_l2318_231804

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l2318_231804


namespace divisibility_by_eleven_l2318_231831

theorem divisibility_by_eleven (n : ℕ) (h : Odd n) : ∃ k : ℤ, (10 : ℤ)^n + 1 = 11 * k := by
  sorry

end divisibility_by_eleven_l2318_231831


namespace solution_set_of_inequality_l2318_231860

theorem solution_set_of_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) (-1/3 : ℝ)).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) (-1/3 : ℝ), (2*y - 1) / (3*y + 1) > 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) (-1/3 : ℝ) → (2*z - 1) / (3*z + 1) ≤ 1) :=
by sorry

end solution_set_of_inequality_l2318_231860


namespace triangle_array_properties_l2318_231842

-- Define what it means to be a triangle array
def is_triangle_array (a b c : ℝ) : Prop :=
  0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

-- Theorem statement
theorem triangle_array_properties 
  (p q r : ℝ) 
  (h : is_triangle_array p q r) : 
  (is_triangle_array (Real.sqrt p) (Real.sqrt q) (Real.sqrt r)) ∧ 
  (∃ p q r : ℝ, is_triangle_array p q r ∧ ¬is_triangle_array (p^2) (q^2) (r^2)) :=
by sorry

end triangle_array_properties_l2318_231842


namespace exponent_multiplication_l2318_231889

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_l2318_231889


namespace geometric_sequence_property_l2318_231827

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Product of the first n terms of a sequence -/
def ProductOfTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (fun acc i => acc * a (i + 1)) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h_geo : GeometricSequence a)
  (h_prop : a (m - 1) * a (m + 1) - 2 * a m = 0)
  (h_product : ProductOfTerms a (2 * m - 1) = 128) :
  m = 4 := by
sorry


end geometric_sequence_property_l2318_231827


namespace max_notebooks_is_eleven_l2318_231898

/-- Represents the maximum number of notebooks that can be purchased with a given budget. -/
def max_notebooks (single_price : ℕ) (pack4_price : ℕ) (pack7_price : ℕ) (budget : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific pricing and budget, the maximum number of notebooks is 11. -/
theorem max_notebooks_is_eleven :
  max_notebooks 2 6 9 15 = 11 := by
  sorry

end max_notebooks_is_eleven_l2318_231898


namespace triangle_inradius_l2318_231884

/-- Given a triangle with perimeter 35 cm and area 78.75 cm², prove its inradius is 4.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 35)
  (h_area : A = 78.75)
  (h_inradius : A = r * p / 2) : 
  r = 4.5 := by
  sorry

end triangle_inradius_l2318_231884


namespace number_problem_l2318_231853

theorem number_problem : 
  ∃ x : ℝ, (1345 - (x / 20.04) = 1295) ∧ (x = 1002) := by sorry

end number_problem_l2318_231853


namespace intersection_of_M_and_N_l2318_231859

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l2318_231859


namespace zero_clever_numbers_l2318_231876

def is_zero_clever (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = a * 1000 + b * 10 + c ∧
    n = (a * 100 + b * 10 + c) * 9 ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem zero_clever_numbers :
  {n : ℕ | is_zero_clever n} = {2025, 4050, 6075} :=
sorry

end zero_clever_numbers_l2318_231876


namespace quadratic_function_property_l2318_231816

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = a * x^2 + b * x + c) 
  (h_cond : f 0 = f 4 ∧ f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
  sorry

end quadratic_function_property_l2318_231816


namespace gcd_840_1764_l2318_231895

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2318_231895


namespace third_cat_weight_calculation_l2318_231836

/-- The weight of the third cat given the weights of the other cats and their average -/
def third_cat_weight (cat1 : ℝ) (cat2 : ℝ) (cat4 : ℝ) (avg_weight : ℝ) : ℝ :=
  4 * avg_weight - (cat1 + cat2 + cat4)

theorem third_cat_weight_calculation :
  third_cat_weight 12 12 9.3 12 = 14.7 := by
  sorry

end third_cat_weight_calculation_l2318_231836


namespace distance_O_to_MN_l2318_231873

/-- The hyperbola C₁: 2x² - y² = 1 -/
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- The ellipse C₂: 4x² + y² = 1 -/
def C₂ (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

/-- M is a point on C₁ -/
def M : ℝ × ℝ := sorry

/-- N is a point on C₂ -/
def N : ℝ × ℝ := sorry

/-- O is the origin -/
def O : ℝ × ℝ := (0, 0)

/-- OM is perpendicular to ON -/
def OM_perp_ON : Prop := sorry

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- The line MN -/
def lineMN : Set (ℝ × ℝ) := sorry

/-- Main theorem: The distance from O to MN is √3/3 -/
theorem distance_O_to_MN :
  C₁ M.1 M.2 → C₂ N.1 N.2 → OM_perp_ON →
  distancePointToLine O lineMN = Real.sqrt 3 / 3 := by sorry

end distance_O_to_MN_l2318_231873


namespace students_studying_all_subjects_l2318_231880

theorem students_studying_all_subjects (total : ℕ) (math : ℕ) (latin : ℕ) (chem : ℕ) 
  (multi : ℕ) (none : ℕ) (h1 : total = 425) (h2 : math = 351) (h3 : latin = 71) 
  (h4 : chem = 203) (h5 : multi = 199) (h6 : none = 8) : 
  ∃ x : ℕ, x = 9 ∧ 
  total - none = math + latin + chem - multi + x := by
  sorry

end students_studying_all_subjects_l2318_231880


namespace divisibility_equivalence_l2318_231856

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) + sequence_a n

theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ sequence_a n ↔ 2^k ∣ n := by
  sorry

end divisibility_equivalence_l2318_231856


namespace right_prism_circumscribed_sphere_radius_l2318_231864

/-- A right prism with a square base -/
structure RightPrism where
  baseEdgeLength : ℝ
  sideEdgeLength : ℝ

/-- The sphere that circumscribes the right prism -/
structure CircumscribedSphere (p : RightPrism) where
  radius : ℝ
  contains_vertices : Prop  -- This represents the condition that all vertices lie on the sphere

/-- Theorem stating that for a right prism with base edge length 1 and side edge length 2,
    if all its vertices lie on a sphere, then the radius of that sphere is √6/2 -/
theorem right_prism_circumscribed_sphere_radius 
  (p : RightPrism) 
  (s : CircumscribedSphere p) 
  (h1 : p.baseEdgeLength = 1) 
  (h2 : p.sideEdgeLength = 2) 
  (h3 : s.contains_vertices) : 
  s.radius = Real.sqrt 6 / 2 := by
sorry

end right_prism_circumscribed_sphere_radius_l2318_231864


namespace kaydence_age_is_twelve_l2318_231867

/-- Represents the ages of family members and the total family age -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  mother_age : ℕ
  brother_age : ℕ
  sister_age : ℕ

/-- Calculates Kaydence's age based on the family's ages -/
def kaydence_age (f : Family) : ℕ :=
  f.total_age - (f.father_age + f.mother_age + f.brother_age + f.sister_age)

/-- Theorem stating that Kaydence's age is 12 given the family conditions -/
theorem kaydence_age_is_twelve :
  ∀ (f : Family),
    f.total_age = 200 →
    f.father_age = 60 →
    f.mother_age = f.father_age - 2 →
    f.brother_age = f.father_age / 2 →
    f.sister_age = 40 →
    kaydence_age f = 12 := by
  sorry

end kaydence_age_is_twelve_l2318_231867


namespace investment_B_is_72000_l2318_231823

/-- Represents the investment and profit distribution in a partnership. -/
structure Partnership where
  investA : ℕ
  investC : ℕ
  profitC : ℕ
  totalProfit : ℕ

/-- Calculates the investment of partner B given the partnership details. -/
def calculateInvestmentB (p : Partnership) : ℕ :=
  p.totalProfit * p.investC / p.profitC - p.investA - p.investC

/-- Theorem stating that given the specified partnership conditions, B's investment is 72000. -/
theorem investment_B_is_72000 (p : Partnership) 
  (h1 : p.investA = 27000)
  (h2 : p.investC = 81000)
  (h3 : p.profitC = 36000)
  (h4 : p.totalProfit = 80000) :
  calculateInvestmentB p = 72000 := by
  sorry

#eval calculateInvestmentB ⟨27000, 81000, 36000, 80000⟩

end investment_B_is_72000_l2318_231823


namespace derivative_at_one_equals_three_l2318_231886

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

-- State the theorem
theorem derivative_at_one_equals_three :
  deriv f 1 = 3 := by
  sorry


end derivative_at_one_equals_three_l2318_231886


namespace students_play_both_football_and_cricket_l2318_231858

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

theorem students_play_both_football_and_cricket :
  students_play_both 450 325 175 50 = 100 := by
  sorry

end students_play_both_football_and_cricket_l2318_231858


namespace natural_number_representation_l2318_231814

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem natural_number_representation (n : ℕ) :
  ∃ x y z : ℕ, n = choose x 1 + choose y 2 + choose z 3 ∧
    ((0 ≤ x ∧ x < y ∧ y < z) ∨ (0 = x ∧ x = y ∧ y < z)) :=
  sorry

end natural_number_representation_l2318_231814


namespace geometric_series_ratio_l2318_231824

theorem geometric_series_ratio (r : ℝ) (h : r ≠ 1) :
  (∀ a : ℝ, a ≠ 0 → a / (1 - r) = 81 * (a * r^4) / (1 - r)) →
  r = 1/3 := by
sorry

end geometric_series_ratio_l2318_231824


namespace ship_distance_theorem_l2318_231868

/-- A function representing the square of the distance of a ship from an island over time. -/
def distance_squared (t : ℝ) : ℝ := 36 * t^2 - 84 * t + 49

/-- The theorem stating the distances at specific times given the initial conditions. -/
theorem ship_distance_theorem :
  (distance_squared 0 = 49) ∧
  (distance_squared 2 = 25) ∧
  (distance_squared 3 = 121) →
  (Real.sqrt (distance_squared 1) = 1) ∧
  (Real.sqrt (distance_squared 4) = 17) := by
  sorry

#check ship_distance_theorem

end ship_distance_theorem_l2318_231868


namespace determine_c_l2318_231834

theorem determine_c (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end determine_c_l2318_231834


namespace smallest_n_with_forty_percent_leftmost_one_l2318_231887

/-- Returns true if the leftmost digit of n is 1 -/
def leftmost_digit_is_one (n : ℕ) : Bool := sorry

/-- Returns the count of numbers from 1 to n (inclusive) with leftmost digit 1 -/
def count_leftmost_one (n : ℕ) : ℕ := sorry

theorem smallest_n_with_forty_percent_leftmost_one :
  ∀ N : ℕ,
    N > 2017 →
    (count_leftmost_one N : ℚ) / N = 2 / 5 →
    N ≥ 1481480 :=
by sorry

end smallest_n_with_forty_percent_leftmost_one_l2318_231887


namespace notebook_duration_example_l2318_231850

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and daily page usage. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end notebook_duration_example_l2318_231850


namespace floor_abs_sum_l2318_231801

theorem floor_abs_sum : ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end floor_abs_sum_l2318_231801


namespace bracket_removal_l2318_231847

theorem bracket_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end bracket_removal_l2318_231847


namespace isosceles_triangles_12_similar_l2318_231851

/-- An isosceles triangle with side ratio 1:2 -/
structure IsoscelesTriangle12 where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  h : a = 2 * b ∨ b = 2 * a  -- Condition for 1:2 ratio

/-- Similarity of isosceles triangles with 1:2 side ratio -/
theorem isosceles_triangles_12_similar (t1 t2 : IsoscelesTriangle12) :
  ∃ (k : ℝ), k > 0 ∧ 
    (t1.a = k * t2.a ∧ t1.b = k * t2.b) ∨
    (t1.a = k * t2.b ∧ t1.b = k * t2.a) :=
sorry

end isosceles_triangles_12_similar_l2318_231851


namespace distinct_polynomials_differ_l2318_231843

-- Define the set X inductively
inductive X : (ℝ → ℝ) → Prop
  | base : X (λ x => x)
  | mul {r} : X r → X (λ x => x * r x)
  | add {r} : X r → X (λ x => x + (1 - x) * r x)

-- Define the theorem
theorem distinct_polynomials_differ (r s : ℝ → ℝ) (hr : X r) (hs : X s) (h_distinct : r ≠ s) :
  ∀ x, 0 < x → x < 1 → r x ≠ s x :=
sorry

end distinct_polynomials_differ_l2318_231843


namespace chicken_crossed_road_l2318_231894

/-- The number of cars dodged by a chicken crossing a road, given the initial and final feather counts. -/
def cars_dodged (initial_feathers final_feathers : ℕ) : ℕ :=
  (initial_feathers - final_feathers) / 2

/-- Theorem stating that the chicken dodged 23 cars given the problem conditions. -/
theorem chicken_crossed_road (initial_feathers final_feathers : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217) :
  cars_dodged initial_feathers final_feathers = 23 := by
  sorry

#eval cars_dodged 5263 5217

end chicken_crossed_road_l2318_231894


namespace lagrange_mean_value_theorem_l2318_231837

theorem lagrange_mean_value_theorem {f : ℝ → ℝ} {a b : ℝ} (hf : Differentiable ℝ f) (hab : a < b) :
  ∃ x₀ ∈ Set.Ioo a b, deriv f x₀ = (f a - f b) / (a - b) := by
  sorry

end lagrange_mean_value_theorem_l2318_231837


namespace coin_flip_problem_l2318_231896

theorem coin_flip_problem (total_flips : ℕ) (tail_head_difference : ℕ) 
  (h1 : total_flips = 211)
  (h2 : tail_head_difference = 81) : 
  ∃ (heads : ℕ), 
    heads + (heads + tail_head_difference) = total_flips ∧ 
    heads = 65 := by
  sorry

end coin_flip_problem_l2318_231896


namespace gcd_lcm_product_30_75_l2318_231870

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end gcd_lcm_product_30_75_l2318_231870


namespace wedding_rsvp_theorem_l2318_231875

def total_guests : ℕ := 200
def yes_percent : ℚ := 83 / 100
def no_percent : ℚ := 9 / 100

theorem wedding_rsvp_theorem :
  (total_guests : ℚ) - (yes_percent * total_guests + no_percent * total_guests) = 16 := by
  sorry

end wedding_rsvp_theorem_l2318_231875


namespace adams_farm_animals_l2318_231892

theorem adams_farm_animals (cows : ℕ) (sheep : ℕ) (pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 := by
sorry

end adams_farm_animals_l2318_231892


namespace parabola_intersection_value_l2318_231820

theorem parabola_intersection_value (m : ℝ) : 
  (m^2 - m - 1 = 0) → (m^2 - m + 2008 = 2009) :=
by sorry

end parabola_intersection_value_l2318_231820


namespace cookies_in_class_l2318_231809

/-- The number of cookies brought by Mona, Jasmine, Rachel, and Carlos -/
def totalCookies (mona jasmine rachel carlos : ℕ) : ℕ :=
  mona + jasmine + rachel + carlos

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class :
  ∀ (mona jasmine rachel carlos : ℕ),
  mona = 20 →
  jasmine = mona - 5 →
  rachel = jasmine + 10 →
  carlos = rachel * 2 →
  totalCookies mona jasmine rachel carlos = 110 := by
  sorry

end cookies_in_class_l2318_231809


namespace train_length_l2318_231840

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 60 → time = 18 → 
  ∃ length : ℝ, abs (length - 300.06) < 0.01 := by
  sorry

end train_length_l2318_231840


namespace inequality_proof_l2318_231815

theorem inequality_proof (a b c : ℝ) : (1/4) * a^2 + b^2 + c^2 ≥ a*b - a*c + 2*b*c := by
  sorry

end inequality_proof_l2318_231815


namespace system_solution_l2318_231863

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (eq2 : x / a + y / b + z / c = a + b + c)
  (eq3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c := by
  sorry

end system_solution_l2318_231863


namespace turkeys_to_ducks_ratio_l2318_231848

/-- Represents the number of birds on Mr. Valentino's farm -/
def total_birds : ℕ := 1800

/-- Represents the number of chickens on Mr. Valentino's farm -/
def chickens : ℕ := 200

/-- Represents the number of ducks on Mr. Valentino's farm -/
def ducks : ℕ := 2 * chickens

/-- Represents the number of turkeys on Mr. Valentino's farm -/
def turkeys : ℕ := total_birds - chickens - ducks

/-- Theorem stating the ratio of turkeys to ducks is 3:1 -/
theorem turkeys_to_ducks_ratio : 
  (turkeys : ℚ) / (ducks : ℚ) = 3 / 1 := by sorry

end turkeys_to_ducks_ratio_l2318_231848


namespace greatest_possible_large_chips_l2318_231888

/-- Represents the number of chips in the box -/
def total_chips : ℕ := 60

/-- Represents the number of large chips -/
def large_chips : ℕ := 29

/-- Represents the number of small chips -/
def small_chips : ℕ := total_chips - large_chips

/-- Represents the difference between small and large chips -/
def difference : ℕ := small_chips - large_chips

theorem greatest_possible_large_chips :
  (total_chips = small_chips + large_chips) ∧
  (∃ p : ℕ, Nat.Prime p ∧ small_chips = large_chips + p ∧ p ∣ large_chips) ∧
  (∀ l : ℕ, l > large_chips →
    ¬(∃ p : ℕ, Nat.Prime p ∧ (total_chips - l) = l + p ∧ p ∣ l)) :=
by sorry

#eval large_chips -- Should output 29
#eval small_chips -- Should output 31
#eval difference -- Should output 2

end greatest_possible_large_chips_l2318_231888


namespace probability_reach_top_correct_l2318_231841

/-- The probability of reaching the top floor using only open doors in a building with n floors and two staircases, where half the doors are randomly locked. -/
def probability_reach_top (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of reaching the top floor using only open doors. -/
theorem probability_reach_top_correct (n : ℕ) (h : n > 1) :
  probability_reach_top n = (2 ^ (n - 1)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end probability_reach_top_correct_l2318_231841


namespace smallest_composite_no_small_factors_l2318_231872

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 15 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  is_composite 289 ∧
  has_no_small_prime_factors 289 ∧
  ∀ m, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m) :=
sorry

end smallest_composite_no_small_factors_l2318_231872


namespace number_fraction_relation_l2318_231810

theorem number_fraction_relation (x : ℝ) (h : (2 / 5) * x = 20) : (1 / 3) * x = 50 / 3 := by
  sorry

end number_fraction_relation_l2318_231810


namespace new_average_age_l2318_231874

/-- Calculates the new average age of a class after a student leaves and the teacher's age is included -/
theorem new_average_age 
  (initial_students : Nat) 
  (initial_average_age : ℝ) 
  (leaving_student_age : ℝ) 
  (teacher_age : ℝ) 
  (h1 : initial_students = 30)
  (h2 : initial_average_age = 10)
  (h3 : leaving_student_age = 11)
  (h4 : teacher_age = 41) : 
  let total_initial_age : ℝ := initial_students * initial_average_age
  let remaining_age : ℝ := total_initial_age - leaving_student_age
  let new_total_age : ℝ := remaining_age + teacher_age
  let new_count : Nat := initial_students
  new_total_age / new_count = 11 := by
  sorry


end new_average_age_l2318_231874


namespace polynomial_division_remainder_l2318_231817

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 19 * X + 53 = (X - 3) * q + 23 :=
by
  sorry

end polynomial_division_remainder_l2318_231817


namespace complex_number_quadrant_l2318_231838

theorem complex_number_quadrant (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (3*m - 2) (m - 1)) : 
  0 < z.re ∧ z.re < 1 ∧ -1/3 < z.im ∧ z.im < 0 :=
by sorry

#check complex_number_quadrant

end complex_number_quadrant_l2318_231838


namespace smallest_sum_of_five_relatively_prime_numbers_l2318_231826

/-- A function that checks if two natural numbers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- A function that checks if a list of natural numbers are pairwise relatively prime -/
def arePairwiseRelativelyPrime (list : List ℕ) : Prop :=
  ∀ (i j : Fin list.length), i.val < j.val → isRelativelyPrime (list.get i) (list.get j)

/-- The main theorem statement -/
theorem smallest_sum_of_five_relatively_prime_numbers :
  ∃ (list : List ℕ),
    list.length = 5 ∧
    arePairwiseRelativelyPrime list ∧
    (∀ (sum : ℕ),
      (∃ (other_list : List ℕ),
        other_list.length = 5 ∧
        arePairwiseRelativelyPrime other_list ∧
        sum = other_list.sum) →
      list.sum ≤ sum) ∧
    list.sum = 4 :=
  sorry

end smallest_sum_of_five_relatively_prime_numbers_l2318_231826


namespace ratio_part_to_whole_l2318_231833

theorem ratio_part_to_whole (N : ℝ) (x : ℝ) 
  (h1 : (1/4) * x * (2/5) * N = 14)
  (h2 : (2/5) * N = 168) : 
  x / N = 2/5 := by
sorry

end ratio_part_to_whole_l2318_231833


namespace tournament_prize_orderings_l2318_231849

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := 5

/-- Represents whether the special reassignment rule is applied -/
def special_rule_applied : Bool := false

/-- Calculates the number of possible outcomes for a single match -/
def outcomes_per_match : ℕ := 2

/-- Theorem stating the number of different prize orderings in the tournament -/
theorem tournament_prize_orderings :
  (outcomes_per_match ^ num_matches : ℕ) = 32 :=
sorry

end tournament_prize_orderings_l2318_231849


namespace triangle_area_product_l2318_231839

/-- Given positive real numbers a and b, and a triangle in the first quadrant
    bounded by the coordinate axes and the line ax + by = 6 with area 6,
    prove that ab = 3. -/
theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ a * x + b * y = 6) →
  ((1 / 2) * (6 / a) * (6 / b) = 6) →
  a * b = 3 := by
sorry


end triangle_area_product_l2318_231839


namespace miller_rabin_composite_detection_probability_l2318_231846

/-- Miller-Rabin test function that returns true if n is probably prime, false if n is definitely composite -/
def miller_rabin_test (n : ℕ) (a : ℕ) : Bool :=
  sorry

/-- The probability that the Miller-Rabin test correctly identifies a composite number -/
def miller_rabin_probability (n : ℕ) : ℝ :=
  sorry

theorem miller_rabin_composite_detection_probability 
  (n : ℕ) (h : ¬ Nat.Prime n) : 
  miller_rabin_probability n ≥ (1/2 : ℝ) :=
sorry

end miller_rabin_composite_detection_probability_l2318_231846


namespace remainder_7n_mod_3_l2318_231879

theorem remainder_7n_mod_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end remainder_7n_mod_3_l2318_231879


namespace farm_feet_count_l2318_231882

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of animals (heads) in the farm -/
def Farm.totalAnimals (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def Farm.totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: In a farm with 44 animals, if there are 24 hens, then the total number of feet is 128 -/
theorem farm_feet_count (f : Farm) (h1 : f.totalAnimals = 44) (h2 : f.hens = 24) : 
  f.totalFeet = 128 := by
  sorry

end farm_feet_count_l2318_231882


namespace infiniteNestedSqrtEqualThree_l2318_231893

/-- The value of the infinite expression sqrt(6 + sqrt(6 + sqrt(6 + ...))) -/
noncomputable def infiniteNestedSqrt : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- Theorem stating that the infinite nested square root equals 3 -/
theorem infiniteNestedSqrtEqualThree : infiniteNestedSqrt = 3 := by
  sorry

end infiniteNestedSqrtEqualThree_l2318_231893


namespace card_game_properties_l2318_231857

/-- A card collection game with 3 colors -/
structure CardGame where
  colors : Nat
  cards_per_color : Nat

/-- The probability of not collecting 3 cards of the same color after 4 purchases -/
def prob_not_three_same (game : CardGame) : ℚ :=
  2 / 3

/-- The distribution of X (number of purchases before collecting 3 cards of the same color) -/
def distribution_X (game : CardGame) (x : Nat) : ℚ :=
  match x with
  | 3 => 1 / 9
  | 4 => 2 / 9
  | 5 => 8 / 27
  | 6 => 20 / 81
  | 7 => 10 / 81
  | _ => 0

/-- The expectation of X -/
def expectation_X (game : CardGame) : ℚ :=
  409 / 81

/-- Main theorem about the card collection game -/
theorem card_game_properties (game : CardGame) 
    (h1 : game.colors = 3) 
    (h2 : game.cards_per_color = 3) : 
  prob_not_three_same game = 2 / 3 ∧ 
  (∀ x, distribution_X game x = match x with
                                | 3 => 1 / 9
                                | 4 => 2 / 9
                                | 5 => 8 / 27
                                | 6 => 20 / 81
                                | 7 => 10 / 81
                                | _ => 0) ∧
  expectation_X game = 409 / 81 := by
  sorry

end card_game_properties_l2318_231857


namespace problem_statement_l2318_231832

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 20)
  (h4 : a*x^4 + b*y^4 = 48)
  (h5 : x + y = -15)
  (h6 : x^2 + y^2 = 55) :
  a*x^5 + b*y^5 = -1065 := by
sorry

end problem_statement_l2318_231832


namespace wiper_generates_sector_l2318_231802

/-- Represents a car wiper -/
structure CarWiper :=
  (length : ℝ)

/-- Represents a windshield -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- Represents a sector on a windshield -/
structure Sector :=
  (angle : ℝ)
  (radius : ℝ)

/-- The action of a car wiper on a windshield -/
def wiper_action (w : CarWiper) (s : Windshield) : Sector :=
  sorry

/-- States that a line (represented by a car wiper) generates a surface (represented by a sector) -/
theorem wiper_generates_sector (w : CarWiper) (s : Windshield) :
  ∃ (sector : Sector), wiper_action w s = sector :=
sorry

end wiper_generates_sector_l2318_231802


namespace angle_DAB_depends_on_triangle_l2318_231803

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the rectangle BCDE
structure Rectangle :=
  (B C D E : ℝ × ℝ)

-- Define the angle β (DAB)
def angle_DAB (tri : Triangle) (rect : Rectangle) : ℝ := sorry

-- State the theorem
theorem angle_DAB_depends_on_triangle (tri : Triangle) (rect : Rectangle) :
  tri.A ≠ tri.B ∧ tri.B ≠ tri.C ∧ tri.C ≠ tri.A →  -- Triangle inequality
  (tri.A.1 - tri.C.1)^2 + (tri.A.2 - tri.C.2)^2 = (tri.B.1 - tri.C.1)^2 + (tri.B.2 - tri.C.2)^2 →  -- CA = CB
  (rect.B = tri.B ∧ rect.C = tri.C) →  -- Rectangle is constructed on CB
  (rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2 > (rect.C.1 - rect.D.1)^2 + (rect.C.2 - rect.D.2)^2 →  -- BC > CD
  ∃ (f : Triangle → ℝ), angle_DAB tri rect = f tri :=
sorry

end angle_DAB_depends_on_triangle_l2318_231803


namespace abs_difference_given_sum_and_product_l2318_231865

theorem abs_difference_given_sum_and_product (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a + b = 6) : 
  |a - b| = 2 * Real.sqrt 6 := by
sorry

end abs_difference_given_sum_and_product_l2318_231865


namespace quadratic_real_roots_condition_l2318_231829

/-- For a quadratic equation x^2 + 4x + k = 0 to have real roots, k must be less than or equal to 4 -/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + k = 0) ↔ k ≤ 4 := by
  sorry

end quadratic_real_roots_condition_l2318_231829


namespace parallel_vectors_x_value_l2318_231891

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (2x, -3),
    and a is parallel to b, prove that x = -3/4 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, -3]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = -3/4 := by
  sorry

end parallel_vectors_x_value_l2318_231891


namespace remaining_clothing_l2318_231828

theorem remaining_clothing (initial : ℕ) (donated_first : ℕ) (thrown_away : ℕ) : 
  initial = 100 →
  donated_first = 5 →
  thrown_away = 15 →
  initial - (donated_first + 3 * donated_first + thrown_away) = 65 := by
  sorry

end remaining_clothing_l2318_231828


namespace min_value_inequality_l2318_231852

theorem min_value_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

end min_value_inequality_l2318_231852


namespace existence_of_unequal_positive_numbers_l2318_231830

theorem existence_of_unequal_positive_numbers : ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b ∧ a + b = a * b := by
  sorry

end existence_of_unequal_positive_numbers_l2318_231830


namespace smallest_d_for_3150_l2318_231845

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem smallest_d_for_3150 : 
  (∃ d : ℕ+, (d.val > 0 ∧ is_perfect_square (3150 * d.val) ∧ 
    ∀ k : ℕ+, k.val > 0 → k.val < d.val → ¬ is_perfect_square (3150 * k.val))) → 
  (∃ d : ℕ+, d.val = 14 ∧ is_perfect_square (3150 * d.val) ∧ 
    ∀ k : ℕ+, k.val > 0 → k.val < d.val → ¬ is_perfect_square (3150 * k.val)) :=
by sorry

end smallest_d_for_3150_l2318_231845


namespace students_on_bus_l2318_231878

theorem students_on_bus (first_stop : ℕ) (second_stop : ℕ) 
  (h1 : first_stop = 39) (h2 : second_stop = 29) :
  first_stop + second_stop = 68 := by
  sorry

end students_on_bus_l2318_231878


namespace abs_fraction_sum_not_one_l2318_231807

theorem abs_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by sorry

end abs_fraction_sum_not_one_l2318_231807


namespace camila_hikes_per_week_l2318_231871

theorem camila_hikes_per_week (camila_hikes : ℕ) (amanda_factor : ℕ) (steven_extra : ℕ) (weeks : ℕ) : 
  camila_hikes = 7 →
  amanda_factor = 8 →
  steven_extra = 15 →
  weeks = 16 →
  ((amanda_factor * camila_hikes + steven_extra - camila_hikes) / weeks : ℚ) = 4 := by
sorry

end camila_hikes_per_week_l2318_231871


namespace line_through_circle_center_l2318_231877

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3*x + y + a = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop := ∀ x y, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5

-- Theorem statement
theorem line_through_circle_center (a : ℝ) : 
  (∃ h k, is_center h k ∧ line_equation h k a) → a = 1 := by
  sorry

end line_through_circle_center_l2318_231877


namespace arithmetic_calculation_l2318_231800

theorem arithmetic_calculation : 1 + 2 * 3 - 4 + 5 = 8 := by
  sorry

end arithmetic_calculation_l2318_231800


namespace product_sqrt_minus_square_eq_1988_l2318_231825

theorem product_sqrt_minus_square_eq_1988 :
  Real.sqrt (1988 * 1989 * 1990 * 1991 + 1) + (-1989^2) = 1988 := by
  sorry

end product_sqrt_minus_square_eq_1988_l2318_231825


namespace smallest_divisible_by_15_l2318_231822

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- The smallest positive integer T consisting only of 0s and 1s that is divisible by 15 -/
def smallest_T : ℕ := sorry

theorem smallest_divisible_by_15 :
  smallest_T > 0 ∧
  only_zero_and_one smallest_T ∧
  smallest_T % 15 = 0 ∧
  smallest_T / 15 = 74 ∧
  ∀ n : ℕ, n > 0 → only_zero_and_one n → n % 15 = 0 → n ≥ smallest_T :=
sorry

end smallest_divisible_by_15_l2318_231822


namespace cylinder_volume_l2318_231897

/-- The volume of a cylinder with base radius 3 and lateral area 12π is 18π. -/
theorem cylinder_volume (r h : ℝ) : r = 3 → 2 * π * r * h = 12 * π → π * r^2 * h = 18 * π := by
  sorry

end cylinder_volume_l2318_231897


namespace unique_m_value_l2318_231861

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ (∀ x ∈ A m, ∀ y ∈ A m, x = y → x = 0 ∨ x = m ∨ x = m^2 - 3*m + 2) :=
  sorry

end unique_m_value_l2318_231861


namespace linear_function_decreases_l2318_231890

/-- A linear function with a negative slope decreases as x increases -/
theorem linear_function_decreases (m b : ℝ) (h : m < 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) := by
  sorry

end linear_function_decreases_l2318_231890


namespace circle_line_theorem_l2318_231881

/-- Two circles passing through a common point -/
structure TwoCircles where
  D1 : ℝ
  E1 : ℝ
  D2 : ℝ
  E2 : ℝ
  h1 : 2^2 + (-1)^2 + D1*2 + E1*(-1) - 3 = 0
  h2 : 2^2 + (-1)^2 + D2*2 + E2*(-1) - 3 = 0

/-- The equation of the line passing through (D1, E1) and (D2, E2) -/
def line_equation (c : TwoCircles) (x y : ℝ) : Prop :=
  2*x - y + 2 = 0

theorem circle_line_theorem (c : TwoCircles) :
  line_equation c c.D1 c.E1 ∧ line_equation c c.D2 c.E2 := by
  sorry

end circle_line_theorem_l2318_231881


namespace seventh_root_unity_product_l2318_231821

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 14 := by
  sorry

end seventh_root_unity_product_l2318_231821


namespace root_transformation_l2318_231854

theorem root_transformation (p q r s : ℂ) : 
  (p^4 - 5*p^2 + 6 = 0) ∧ 
  (q^4 - 5*q^2 + 6 = 0) ∧ 
  (r^4 - 5*r^2 + 6 = 0) ∧ 
  (s^4 - 5*s^2 + 6 = 0) →
  ((p+q)/(r+s))^4 + 4*((p+q)/(r+s))^3 + 6*((p+q)/(r+s))^2 + 4*((p+q)/(r+s)) + 1 = 0 ∧
  ((p+r)/(q+s))^4 + 4*((p+r)/(q+s))^3 + 6*((p+r)/(q+s))^2 + 4*((p+r)/(q+s)) + 1 = 0 ∧
  ((p+s)/(q+r))^4 + 4*((p+s)/(q+r))^3 + 6*((p+s)/(q+r))^2 + 4*((p+s)/(q+r)) + 1 = 0 ∧
  ((q+r)/(p+s))^4 + 4*((q+r)/(p+s))^3 + 6*((q+r)/(p+s))^2 + 4*((q+r)/(p+s)) + 1 = 0 :=
by sorry

end root_transformation_l2318_231854


namespace bears_permutations_l2318_231813

theorem bears_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end bears_permutations_l2318_231813


namespace min_a_value_l2318_231811

open Real

-- Define the function f(x) = x/ln(x) - 1/(4x)
noncomputable def f (x : ℝ) : ℝ := x / log x - 1 / (4 * x)

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∃ x ∈ Set.Icc (exp 1) (exp 2), x / log x ≤ 1/4 + a*x) ↔ 
  a ≥ 1/2 - 1/(4 * (exp 2)^2) :=
sorry

end min_a_value_l2318_231811


namespace paint_canvas_cost_ratio_l2318_231806

theorem paint_canvas_cost_ratio 
  (canvas_original : ℝ) 
  (paint_original : ℝ) 
  (canvas_decrease : ℝ) 
  (paint_decrease : ℝ) 
  (total_decrease : ℝ)
  (h1 : canvas_decrease = 0.4)
  (h2 : paint_decrease = 0.6)
  (h3 : total_decrease = 0.5599999999999999)
  (h4 : canvas_original > 0)
  (h5 : paint_original > 0)
  (h6 : (1 - paint_decrease) * paint_original + (1 - canvas_decrease) * canvas_original 
      = (1 - total_decrease) * (paint_original + canvas_original)) :
  paint_original / canvas_original = 4 := by
sorry

end paint_canvas_cost_ratio_l2318_231806


namespace smallest_square_box_for_cards_l2318_231819

/-- Represents the dimensions of a business card -/
structure BusinessCard where
  width : ℕ
  length : ℕ

/-- Represents a square box -/
structure SquareBox where
  side : ℕ

/-- Checks if a square box can fit a whole number of business cards without overlapping -/
def canFitCards (box : SquareBox) (card : BusinessCard) : Prop :=
  (box.side % card.width = 0) ∧ (box.side % card.length = 0)

/-- Theorem: The smallest square box that can fit business cards of 5x7 cm has sides of 35 cm -/
theorem smallest_square_box_for_cards :
  let card := BusinessCard.mk 5 7
  let box := SquareBox.mk 35
  (canFitCards box card) ∧
  (∀ (smallerBox : SquareBox), smallerBox.side < box.side → ¬(canFitCards smallerBox card)) :=
by sorry

end smallest_square_box_for_cards_l2318_231819


namespace repeating_decimal_value_l2318_231844

def repeating_decimal : ℚ := 33 / 99999

theorem repeating_decimal_value : 
  (10^5 - 10^3 : ℚ) * repeating_decimal = 32.67 := by sorry

end repeating_decimal_value_l2318_231844


namespace curve_properties_l2318_231862

-- Define the curve C
def C (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the condition k ≠ -1
def k_not_neg_one (k : ℝ) : Prop := k ≠ -1

theorem curve_properties (k : ℝ) (h : k_not_neg_one k) :
  -- 1. C is always a circle
  (∃ (center_x center_y radius : ℝ), ∀ (x y : ℝ),
    C k x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
  -- The centers of the circles lie on the line y = 2x - 5
  (∃ (center_x center_y : ℝ), 
    (∀ (x y : ℝ), C k x y → (x - center_x)^2 + (y - center_y)^2 = (5*(k+1)^2)) ∧
    center_y = 2*center_x - 5) ∧
  -- 2. C passes through the fixed point (1, -3)
  C k 1 (-3) ∧
  -- 3. When C is tangent to the x-axis, k = 5 ± 3√5
  (∃ (x : ℝ), C k x 0 ∧ 
    (∀ (y : ℝ), y ≠ 0 → ¬(C k x y)) →
    (k = 5 + 3*Real.sqrt 5 ∨ k = 5 - 3*Real.sqrt 5)) :=
by sorry

end curve_properties_l2318_231862


namespace headphone_price_reduction_l2318_231835

theorem headphone_price_reduction (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ) :
  original_price = 120 →
  first_discount_rate = 0.25 →
  second_discount_rate = 0.1 →
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 81 := by
sorry

end headphone_price_reduction_l2318_231835


namespace height_of_equilateral_triangle_l2318_231808

/-- An equilateral triangle with base 2 and an inscribed circle --/
structure EquilateralTriangleWithInscribedCircle where
  /-- The base of the triangle --/
  base : ℝ
  /-- The height of the triangle --/
  height : ℝ
  /-- The radius of the inscribed circle --/
  radius : ℝ
  /-- The base is 2 --/
  base_eq_two : base = 2
  /-- The radius is half the height --/
  radius_half_height : radius = height / 2

/-- The height of an equilateral triangle with base 2 and an inscribed circle is √3 --/
theorem height_of_equilateral_triangle
  (triangle : EquilateralTriangleWithInscribedCircle) :
  triangle.height = Real.sqrt 3 := by
  sorry

end height_of_equilateral_triangle_l2318_231808


namespace hundred_billion_scientific_notation_l2318_231818

theorem hundred_billion_scientific_notation :
  (100000000000 : ℕ) = 1 * 10^11 :=
sorry

end hundred_billion_scientific_notation_l2318_231818


namespace inequality_iff_p_in_unit_interval_l2318_231805

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The proposition that pf(x) + qf(y) ≥ f(px + qy) for all real x, y -/
def inequality_holds (a b p q : ℝ) : Prop :=
  ∀ x y : ℝ, p * f a b x + q * f a b y ≥ f a b (p*x + q*y)

theorem inequality_iff_p_in_unit_interval (a b : ℝ) :
  ∀ p q : ℝ, p + q = 1 →
    (inequality_holds a b p q ↔ 0 ≤ p ∧ p ≤ 1) :=
by sorry

end inequality_iff_p_in_unit_interval_l2318_231805


namespace siblings_selection_probability_l2318_231855

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_ritu : ℚ) 
  (h_ram : p_ram = 3 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_ritu : p_ritu = 2 / 9) : 
  p_ram * p_ravi * p_ritu = 2 / 105 := by
  sorry

end siblings_selection_probability_l2318_231855
