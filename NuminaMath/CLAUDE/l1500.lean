import Mathlib

namespace min_yz_minus_xy_l1500_150004

/-- Represents a triangle with integer side lengths -/
structure Triangle :=
  (xy yz xz : ℕ)

/-- The perimeter of the triangle -/
def Triangle.perimeter (t : Triangle) : ℕ := t.xy + t.yz + t.xz

/-- Predicate for a valid triangle satisfying the given conditions -/
def isValidTriangle (t : Triangle) : Prop :=
  t.xy < t.yz ∧ t.yz ≤ t.xz ∧
  t.perimeter = 2010 ∧
  t.xy + t.yz > t.xz ∧ t.xy + t.xz > t.yz ∧ t.yz + t.xz > t.xy

theorem min_yz_minus_xy (t : Triangle) (h : isValidTriangle t) :
  ∀ (t' : Triangle), isValidTriangle t' → t'.yz - t'.xy ≥ 1 :=
sorry

end min_yz_minus_xy_l1500_150004


namespace bacteria_growth_example_l1500_150019

/-- The time needed for bacteria to reach a certain population -/
def bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) : ℕ :=
  let growth_cycles := (final_count / initial_count).log growth_factor
  growth_cycles * growth_time

/-- Theorem: The time needed for 200 bacteria to reach 145,800 bacteria, 
    given that they triple every 5 hours, is 30 hours. -/
theorem bacteria_growth_example : bacteria_growth_time 200 145800 3 5 = 30 := by
  sorry

end bacteria_growth_example_l1500_150019


namespace parabola_properties_l1500_150016

-- Define the parabola
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 2 * a * x

-- Define the properties of the parabola C
def C : Parabola where
  a := 1  -- This makes the equation y² = 2x

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (2,2)
  C.equation 2 2 ∧
  -- The focus is on the x-axis at (1/2, 0)
  C.equation (1/2) 0 ∧
  -- The intersection with x - y - 1 = 0 gives |MN| = 2√6
  ∃ (x₁ x₂ : ℝ),
    C.equation x₁ (x₁ - 1) ∧
    C.equation x₂ (x₂ - 1) ∧
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 + ((x₂ - 1) - (x₁ - 1))^2 = 24 := by
  sorry

end parabola_properties_l1500_150016


namespace initially_calculated_average_of_class_l1500_150010

/-- The initially calculated average height of a class of boys -/
def initially_calculated_average (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) : ℚ :=
  actual_average + (initial_error : ℚ) / num_boys

/-- Theorem stating the initially calculated average height -/
theorem initially_calculated_average_of_class (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) 
  (h1 : num_boys = 35)
  (h2 : actual_average = 178)
  (h3 : initial_error = 50) :
  initially_calculated_average num_boys actual_average initial_error = 179 + 3 / 7 := by
  sorry

end initially_calculated_average_of_class_l1500_150010


namespace tara_ice_cream_yoghurt_spending_l1500_150073

theorem tara_ice_cream_yoghurt_spending :
  let ice_cream_cartons : ℕ := 19
  let yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 7
  let yoghurt_price : ℕ := 1
  let ice_cream_total : ℕ := ice_cream_cartons * ice_cream_price
  let yoghurt_total : ℕ := yoghurt_cartons * yoghurt_price
  ice_cream_total - yoghurt_total = 129 :=
by sorry

end tara_ice_cream_yoghurt_spending_l1500_150073


namespace inscribed_circle_triangle_sides_l1500_150000

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- The length of the first segment on one side --/
  s1 : ℝ
  /-- The length of the second segment on one side --/
  s2 : ℝ
  /-- The length of the second side --/
  a : ℝ
  /-- The length of the third side --/
  b : ℝ
  /-- Ensure all lengths are positive --/
  r_pos : r > 0
  s1_pos : s1 > 0
  s2_pos : s2 > 0
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem about a specific triangle with an inscribed circle --/
theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle)
  (h1 : t.r = 4)
  (h2 : t.s1 = 6)
  (h3 : t.s2 = 8) :
  t.a = 13 ∧ t.b = 15 := by
  sorry


end inscribed_circle_triangle_sides_l1500_150000


namespace football_team_progress_l1500_150095

/-- Given a football team's yard changes, calculate their progress -/
def teamProgress (lost : ℤ) (gained : ℤ) : ℤ :=
  gained - lost

theorem football_team_progress :
  let lost : ℤ := 5
  let gained : ℤ := 7
  teamProgress lost gained = 2 := by
  sorry

end football_team_progress_l1500_150095


namespace simple_interest_calculation_l1500_150013

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.05)
  (h3 : time = 1) :
  principal * rate * time = 500 := by
sorry

end simple_interest_calculation_l1500_150013


namespace cube_surface_area_from_volume_l1500_150082

-- Define the volume of the cube
def cube_volume : ℝ := 125

-- Theorem stating the relationship between volume and surface area of one side
theorem cube_surface_area_from_volume :
  ∃ (side_length : ℝ), 
    side_length^3 = cube_volume ∧ 
    side_length^2 = 25 :=
by sorry

end cube_surface_area_from_volume_l1500_150082


namespace planes_perpendicular_from_parallel_lines_l1500_150068

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_perpendicular_from_parallel_lines
  (m n : Line3D) (α β : Plane3D)
  (h1 : parallel m n)
  (h2 : contained_in m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_plane_plane α β :=
sorry

end planes_perpendicular_from_parallel_lines_l1500_150068


namespace square_perimeter_l1500_150005

/-- Given a square with area 625 cm², prove its perimeter is 100 cm -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end square_perimeter_l1500_150005


namespace negative_number_identification_l1500_150099

theorem negative_number_identification :
  let numbers : List ℚ := [1, 0, 1/2, -2]
  ∀ x ∈ numbers, x < 0 ↔ x = -2 :=
by sorry

end negative_number_identification_l1500_150099


namespace sqrt_equation_solutions_l1500_150045

theorem sqrt_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 - x) + Real.sqrt x
  ∀ x : ℝ, f x = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

end sqrt_equation_solutions_l1500_150045


namespace bucket_weight_l1500_150093

/-- Given a bucket with unknown empty weight and full water weight,
    if it weighs c kilograms when three-quarters full and b kilograms when half-full,
    then its weight when one-third full is (5/3)b - (2/3)c kilograms. -/
theorem bucket_weight (b c : ℝ) : 
  (∃ x y : ℝ, x + 3/4 * y = c ∧ x + 1/2 * y = b) → 
  (∃ z : ℝ, z = 5/3 * b - 2/3 * c ∧ 
    ∀ x y : ℝ, x + 3/4 * y = c → x + 1/2 * y = b → x + 1/3 * y = z) :=
by sorry

end bucket_weight_l1500_150093


namespace inequality_proof_l1500_150097

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c) := by
  sorry

end inequality_proof_l1500_150097


namespace inequality_equivalence_l1500_150071

theorem inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < -6 ↔ x ∈ Set.Ioo (-9/2 : ℝ) 1 := by
  sorry

end inequality_equivalence_l1500_150071


namespace cube_root_square_l1500_150074

theorem cube_root_square (y : ℝ) : (y + 5) ^ (1/3 : ℝ) = 3 → (y + 5)^2 = 729 := by
  sorry

end cube_root_square_l1500_150074


namespace hari_contribution_is_2160_l1500_150078

/-- Represents the investment details and profit sharing ratio --/
structure InvestmentDetails where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution given the investment details --/
def calculate_hari_contribution (details : InvestmentDetails) : ℕ :=
  (details.praveen_investment * details.praveen_months * details.profit_ratio_hari) /
  (details.profit_ratio_praveen * details.hari_months)

/-- Theorem stating that Hari's contribution is 2160 given the problem conditions --/
theorem hari_contribution_is_2160 :
  let details : InvestmentDetails := {
    praveen_investment := 3360,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_contribution details = 2160 := by
  sorry

#eval calculate_hari_contribution {
  praveen_investment := 3360,
  praveen_months := 12,
  hari_months := 7,
  total_months := 12,
  profit_ratio_praveen := 2,
  profit_ratio_hari := 3
}

end hari_contribution_is_2160_l1500_150078


namespace extreme_value_and_monotonicity_l1500_150009

/-- The function f(x) = x³ + ax² + bx + a² -/
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_and_monotonicity (a b : ℝ) :
  (f 1 a b = 10 ∧ f' 1 a b = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x : ℝ, 
    (b = -a^2 → 
      (a > 0 → 
        ((x < -a ∨ x > a/3) → f' x a (-a^2) > 0) ∧
        ((-a < x ∧ x < a/3) → f' x a (-a^2) < 0)) ∧
      (a < 0 → 
        ((x < a/3 ∨ x > -a) → f' x a (-a^2) > 0) ∧
        ((a/3 < x ∧ x < -a) → f' x a (-a^2) < 0)) ∧
      (a = 0 → f' x a (-a^2) > 0))) :=
by sorry

end extreme_value_and_monotonicity_l1500_150009


namespace set_a_equals_set_b_l1500_150098

/-- A positive integer that is not a perfect square -/
structure NonSquare (a : ℕ) : Prop where
  pos : 0 < a
  not_square : ∀ n : ℕ, n^2 ≠ a

/-- The equation k = (x^2 - a) / (x^2 - y^2) has a solution in ℤ^2 -/
def HasSolution (k a : ℕ) : Prop :=
  ∃ x y : ℤ, k = (x^2 - a) / (x^2 - y^2)

/-- The set of positive integers k for which the equation has a solution with x > √a -/
def SetA (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, x^2 > a ∧ HasSolution k a}

/-- The set of positive integers k for which the equation has a solution with 0 ≤ x < √a -/
def SetB (a : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ ∃ x y : ℤ, 0 ≤ x^2 ∧ x^2 < a ∧ HasSolution k a}

/-- The main theorem: Set A equals Set B for any non-square positive integer a -/
theorem set_a_equals_set_b (a : ℕ) (h : NonSquare a) : SetA a = SetB a := by
  sorry

end set_a_equals_set_b_l1500_150098


namespace band_members_proof_l1500_150042

/-- Represents the price per set of costumes based on the quantity purchased -/
def price_per_set (quantity : ℕ) : ℕ :=
  if quantity ≤ 39 then 80
  else if quantity ≤ 79 then 70
  else 60

theorem band_members_proof :
  ∀ (x y : ℕ),
    x + y = 75 →
    x ≥ 40 →
    price_per_set x * x + price_per_set y * y = 5600 →
    x = 40 ∧ y = 35 := by
  sorry

end band_members_proof_l1500_150042


namespace students_at_start_correct_l1500_150081

/-- The number of students at the start of the year in fourth grade -/
def students_at_start : ℕ := 10

/-- The number of students added during the year -/
def students_added : ℝ := 4.0

/-- The number of new students who came to school -/
def new_students : ℝ := 42.0

/-- The total number of students at the end of the year -/
def students_at_end : ℕ := 56

/-- Theorem stating that the number of students at the start of the year is correct -/
theorem students_at_start_correct :
  students_at_start + (students_added + new_students) = students_at_end := by
  sorry

end students_at_start_correct_l1500_150081


namespace system_solution_l1500_150060

/-- The function φ(t) = 2t^3 + t - 2 -/
def φ (t : ℝ) : ℝ := 2 * t^3 + t - 2

/-- The system of equations -/
def satisfies_system (x y z : ℝ) : Prop :=
  x^5 = φ y ∧ y^5 = φ z ∧ z^5 = φ x

theorem system_solution (x y z : ℝ) (h : satisfies_system x y z) :
  x = y ∧ y = z ∧ φ x = x^5 := by
  sorry

#check system_solution

end system_solution_l1500_150060


namespace paint_mixture_ratio_l1500_150046

/-- Given a paint mixture with ratio blue:green:white as 5:3:7,
    prove that using 21 quarts of white paint requires 9 quarts of green paint. -/
theorem paint_mixture_ratio (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : green = 9 := by
  sorry

end paint_mixture_ratio_l1500_150046


namespace circle_symmetry_l1500_150084

/-- Given a circle with equation x^2 + y^2 + 2x - 4y + 4 = 0 that is symmetric about the line y = 2x + b, prove that b = 4 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 4 = 0 → 
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 4 = 0 ∧ 
    y' = 2*x' + b ∧ 
    (x - x')^2 + (y - y')^2 = (x - x')^2 + ((2*x + b) - (2*x' + b))^2) →
  b = 4 := by
sorry

end circle_symmetry_l1500_150084


namespace paperclip_theorem_l1500_150008

/-- The day of the week when Jasmine first has more than 500 paperclips -/
theorem paperclip_theorem : ∃ k : ℕ, k > 0 ∧ 
  (∀ j : ℕ, j < k → 5 * 3^j ≤ 500) ∧ 
  5 * 3^k > 500 ∧
  k = 5 := by
  sorry

end paperclip_theorem_l1500_150008


namespace hash_2_3_neg1_l1500_150085

def hash (a b c : ℝ) : ℝ := b^3 - 4*a*c + b

theorem hash_2_3_neg1 : hash 2 3 (-1) = 38 := by
  sorry

end hash_2_3_neg1_l1500_150085


namespace max_boxes_per_delivery_l1500_150044

/-- Represents the maximum capacity of each truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the weight of a light box in pounds -/
def lightBoxWeight : ℕ := 10

/-- Represents the weight of a heavy box in pounds -/
def heavyBoxWeight : ℕ := 40

/-- Represents the number of trucks available for each delivery -/
def numberOfTrucks : ℕ := 3

/-- Theorem stating the maximum number of boxes that can be shipped in each delivery -/
theorem max_boxes_per_delivery :
  ∃ (n : ℕ), n = numberOfTrucks * truckCapacity / (lightBoxWeight + heavyBoxWeight) * 2 ∧ n = 240 := by
  sorry

end max_boxes_per_delivery_l1500_150044


namespace mango_problem_l1500_150050

theorem mango_problem (alexis dilan ashley : ℕ) : 
  alexis = 4 * (dilan + ashley) →
  ashley = 2 * dilan →
  alexis = 60 →
  alexis + dilan + ashley = 75 :=
by
  sorry

end mango_problem_l1500_150050


namespace smallest_with_twelve_divisors_l1500_150077

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_twelve_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by sorry

end smallest_with_twelve_divisors_l1500_150077


namespace hannah_friday_distance_l1500_150049

/-- The distance Hannah ran on Monday in kilometers -/
def monday_km : ℝ := 9

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_m : ℝ := 4816

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def additional_m : ℝ := 2089

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem hannah_friday_distance :
  ∃ (friday_m : ℝ),
    (monday_km * km_to_m = wednesday_m + friday_m + additional_m) ∧
    friday_m = 2095 := by
  sorry

end hannah_friday_distance_l1500_150049


namespace cricket_average_l1500_150029

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 79 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 35 := by
sorry

end cricket_average_l1500_150029


namespace differential_y_differential_F_differential_z_dz_at_zero_l1500_150037

noncomputable section

-- Function definitions
def y (x : ℝ) := x^3 - 3^x
def F (φ : ℝ) := Real.cos (φ/3) + Real.sin (3/φ)
def z (x : ℝ) := Real.log (1 + Real.exp (10*x)) + Real.arctan (Real.exp (5*x))⁻¹

-- Theorem statements
theorem differential_y (x : ℝ) :
  deriv y x = 3*x^2 - 3^x * Real.log 3 :=
sorry

theorem differential_F (φ : ℝ) (h : φ ≠ 0) :
  deriv F φ = -1/3 * Real.sin (φ/3) - 3 * Real.cos (3/φ) / φ^2 :=
sorry

theorem differential_z (x : ℝ) :
  deriv z x = (5 * Real.exp (5*x) * (2 * Real.exp (5*x) - 1)) / (1 + Real.exp (10*x)) :=
sorry

theorem dz_at_zero :
  (deriv z 0) * 0.1 = 0.25 :=
sorry

end

end differential_y_differential_F_differential_z_dz_at_zero_l1500_150037


namespace correct_statements_are_ACD_l1500_150090

-- Define the set of all statements
inductive Statement : Type
| A : Statement
| B : Statement
| C : Statement
| D : Statement

-- Define a function to check if a statement is correct
def is_correct : Statement → Prop
| Statement.A => ∀ (residual_width : ℝ) (fitting_quality : ℝ),
    residual_width < 0 → fitting_quality > 0
| Statement.B => ∀ (r_A r_B : ℝ),
    r_A = 0.97 ∧ r_B = -0.99 → abs r_A > abs r_B
| Statement.C => ∀ (R_squared fitting_quality : ℝ),
    R_squared < 0 → fitting_quality < 0
| Statement.D => ∀ (n k d : ℕ),
    n = 10 ∧ k = 2 ∧ d = 3 →
    (Nat.choose d 1 * Nat.choose (n - d) (k - 1)) / Nat.choose n k = 7 / 15

-- Define the set of correct statements
def correct_statements : Set Statement :=
  {s | is_correct s}

-- Theorem to prove
theorem correct_statements_are_ACD :
  correct_statements = {Statement.A, Statement.C, Statement.D} :=
sorry

end correct_statements_are_ACD_l1500_150090


namespace gcf_40_48_l1500_150020

theorem gcf_40_48 : Nat.gcd 40 48 = 8 := by
  sorry

end gcf_40_48_l1500_150020


namespace inequality_proof_l1500_150089

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2018 + b^2018)^2019 > (a^2019 + b^2019)^2018 := by
  sorry

end inequality_proof_l1500_150089


namespace range_of_h_sign_and_inequality_smallest_upper_bound_l1500_150062

-- Define sets A and M_n
def A : Set (ℝ → ℝ) := {f | ∃ k, ∀ x > 0, f x < k}
def M (n : ℕ) : Set (ℝ → ℝ) := {f | ∀ x y, 0 < x ∧ x < y → (f x / x^n) < (f y / y^n)}

-- Statement 1
theorem range_of_h (h : ℝ) :
  (fun x => x^3 + h) ∈ M 1 ↔ h ≤ 0 :=
sorry

-- Statement 2
theorem sign_and_inequality (f : ℝ → ℝ) (a b d : ℝ) 
  (hf : f ∈ M 1) (hab : 0 < a ∧ a < b) (hd : f a = d ∧ f b = d) :
  d < 0 ∧ f (a + b) > 2 * d :=
sorry

-- Statement 3
theorem smallest_upper_bound (m : ℝ) :
  (∀ f ∈ A ∩ M 2, ∀ x > 0, f x < m) ↔ m ≥ 0 :=
sorry

end range_of_h_sign_and_inequality_smallest_upper_bound_l1500_150062


namespace cube_packing_percentage_l1500_150079

/-- Calculates the number of whole cubes that can fit along a dimension -/
def cubesFit (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the volume of a rectangular box -/
def boxVolume (length width height : ℕ) : ℕ :=
  length * width * height

/-- Calculates the volume of a cube -/
def cubeVolume (size : ℕ) : ℕ :=
  size * size * size

/-- Theorem: The percentage of volume occupied by 4-inch cubes in a box with 
    dimensions 8 × 5 × 14 inches is 24/35 * 100% -/
theorem cube_packing_percentage :
  let boxLength : ℕ := 8
  let boxWidth : ℕ := 5
  let boxHeight : ℕ := 14
  let cubeSize : ℕ := 4
  let cubesAlongLength := cubesFit boxLength cubeSize
  let cubesAlongWidth := cubesFit boxWidth cubeSize
  let cubesAlongHeight := cubesFit boxHeight cubeSize
  let totalCubes := cubesAlongLength * cubesAlongWidth * cubesAlongHeight
  let volumeOccupied := totalCubes * cubeVolume cubeSize
  let totalVolume := boxVolume boxLength boxWidth boxHeight
  (volumeOccupied : ℚ) / totalVolume * 100 = 24 / 35 * 100 := by
  sorry

end cube_packing_percentage_l1500_150079


namespace clownfish_in_display_tank_l1500_150023

theorem clownfish_in_display_tank 
  (total_fish : ℕ)
  (clownfish blowfish : ℕ)
  (blowfish_in_own_tank : ℕ)
  (h1 : total_fish = 100)
  (h2 : clownfish = blowfish)
  (h3 : blowfish_in_own_tank = 26)
  (h4 : total_fish = clownfish + blowfish) :
  let blowfish_in_display := blowfish - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  let final_clownfish_in_display := initial_clownfish_in_display - initial_clownfish_in_display / 3
  final_clownfish_in_display = 16 := by
sorry

end clownfish_in_display_tank_l1500_150023


namespace eq_length_is_40_l1500_150054

/-- Represents a trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  -- Lengths of the trapezoid sides
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  -- Ensure EF is parallel to GH (represented by their lengths being different)
  ef_parallel_gh : ef ≠ gh
  -- Circle center Q is on EF
  eq : ℝ
  -- Circle is tangent to FG and HE (implicitly assumed by the structure)

/-- The specific trapezoid from the problem -/
def problemTrapezoid : InscribedCircleTrapezoid where
  ef := 100
  fg := 60
  gh := 22
  he := 80
  ef_parallel_gh := by norm_num
  eq := 40  -- This is what we want to prove

/-- The main theorem: EQ = 40 in the given trapezoid -/
theorem eq_length_is_40 : problemTrapezoid.eq = 40 := by
  sorry

#eval problemTrapezoid.eq  -- Should output 40

end eq_length_is_40_l1500_150054


namespace angle_theta_trig_values_l1500_150027

/-- An angle θ with vertex at the origin, initial side along positive x-axis, and terminal side on y = 2x -/
structure AngleTheta where
  terminal_side : ∀ (x y : ℝ), y = 2 * x

theorem angle_theta_trig_values (θ : AngleTheta) :
  ∃ (s c : ℝ),
    s^2 + c^2 = 1 ∧
    |s| = 2 * Real.sqrt 5 / 5 ∧
    |c| = Real.sqrt 5 / 5 ∧
    s / c = 2 := by
  sorry

end angle_theta_trig_values_l1500_150027


namespace third_number_value_l1500_150001

theorem third_number_value : ∃ x : ℝ, 3 + 33 + x + 3.33 = 369.63 ∧ x = 330.30 := by
  sorry

end third_number_value_l1500_150001


namespace apple_boxes_count_apple_boxes_count_specific_l1500_150058

theorem apple_boxes_count (apples_per_crate : ℕ) (crates_delivered : ℕ) 
  (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  let total_apples := apples_per_crate * crates_delivered
  let remaining_apples := total_apples - rotten_apples
  remaining_apples / apples_per_box

theorem apple_boxes_count_specific : 
  apple_boxes_count 180 12 160 20 = 100 := by
  sorry

end apple_boxes_count_apple_boxes_count_specific_l1500_150058


namespace A_3_2_l1500_150021

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 12 := by sorry

end A_3_2_l1500_150021


namespace sprint_race_losing_distance_l1500_150057

/-- Represents a sprint race between Kelly and Abel -/
structure SprintRace where
  raceLength : ℝ
  headStart : ℝ
  extraDistanceToOvertake : ℝ

/-- Calculates the distance by which Abel lost the race to Kelly -/
def losingDistance (race : SprintRace) : ℝ :=
  race.headStart + race.extraDistanceToOvertake

theorem sprint_race_losing_distance : 
  let race : SprintRace := {
    raceLength := 100,
    headStart := 3,
    extraDistanceToOvertake := 19.9
  }
  losingDistance race = 22.9 := by sorry

end sprint_race_losing_distance_l1500_150057


namespace divisibility_by_30_l1500_150011

theorem divisibility_by_30 :
  (∃ p : ℕ, p.Prime ∧ p ≥ 7 ∧ 30 ∣ (p^2 - 1)) ∧
  (∃ q : ℕ, q.Prime ∧ q ≥ 7 ∧ ¬(30 ∣ (q^2 - 1))) :=
by sorry

end divisibility_by_30_l1500_150011


namespace custom_op_example_l1500_150041

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem custom_op_example : custom_op 4 2 = 24 := by
  sorry

end custom_op_example_l1500_150041


namespace max_consecutive_sum_l1500_150072

/-- The sum of n consecutive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + (n - 1)) / 2

/-- The maximum number of consecutive positive integers starting from 3 
    that can be added together before the sum exceeds 500 -/
theorem max_consecutive_sum : 
  (∀ m : ℕ, m ≤ 29 → consecutiveSum m 3 ≤ 500) ∧ 
  consecutiveSum 30 3 > 500 := by
  sorry

end max_consecutive_sum_l1500_150072


namespace circle_intersection_theorem_specific_m_value_diameter_circle_equation_l1500_150039

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 5 :=
sorry

theorem specific_m_value :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  (8/5 : ℝ) = 8/5 :=
sorry

theorem diameter_circle_equation :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  ∀ x y : ℝ,
  x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
  (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 :=
sorry

end circle_intersection_theorem_specific_m_value_diameter_circle_equation_l1500_150039


namespace sphere_volume_equals_surface_area_l1500_150075

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end sphere_volume_equals_surface_area_l1500_150075


namespace inequality_proof_l1500_150002

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := by
  sorry

end inequality_proof_l1500_150002


namespace divisibility_conditions_l1500_150018

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, 3 ∣ (5*n^2 + 10*n + 8) ↔ n = 2 + 3*k) ∧
  (∃ k : ℤ, 4 ∣ (5*n^2 + 10*n + 8) ↔ n = 2*k) := by
  sorry

end divisibility_conditions_l1500_150018


namespace circle_sections_theorem_l1500_150047

-- Define the circle and its sections
def Circle (r : ℝ) := { x : ℝ × ℝ | x.1^2 + x.2^2 = r^2 }

structure Section (r : ℝ) where
  area : ℝ
  perimeter : ℝ

-- Define the theorem
theorem circle_sections_theorem (r : ℝ) (h : r > 0) :
  ∃ (s1 s2 s3 : Section r),
    -- Areas are equal and sum to the circle's area
    s1.area = s2.area ∧ s2.area = s3.area ∧
    s1.area + s2.area + s3.area = π * r^2 ∧
    -- Each section's area is r²π/3
    s1.area = (π * r^2) / 3 ∧
    -- Perimeters are equal to the circle's perimeter
    s1.perimeter = s2.perimeter ∧ s2.perimeter = s3.perimeter ∧
    s1.perimeter = 2 * π * r :=
by
  sorry


end circle_sections_theorem_l1500_150047


namespace economy_relationship_l1500_150007

/-- Given an economy with product X, price P, and total cost C, prove the relationship
    between these variables and calculate specific values. -/
theorem economy_relationship (k k' : ℝ) : 
  (∀ (X P : ℝ), X * P = k) →  -- X is inversely proportional to P
  (200 : ℝ) * 10 = k →        -- When P = 10, X = 200
  (∀ (C X : ℝ), C = k' * X) → -- C is directly proportional to X
  4000 = k' * 200 →           -- When X = 200, C = 4000
  (∃ (X C : ℝ), X * 50 = k ∧ C = k' * X ∧ X = 40 ∧ C = 800) := by
sorry

end economy_relationship_l1500_150007


namespace special_cone_volume_l1500_150048

/-- A cone with circumscribed and inscribed spheres sharing the same center -/
structure SpecialCone where
  /-- The radius of the circumscribed sphere -/
  r_circum : ℝ
  /-- The circumscribed and inscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of the special cone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of the special cone is 3π when the radius of the circumscribed sphere is 2 -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.r_circum = 2) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 3 * Real.pi := by sorry

end special_cone_volume_l1500_150048


namespace dice_roll_probability_l1500_150061

/-- The probability of rolling a number less than four on a six-sided die -/
def prob_first_die : ℚ := 1 / 2

/-- The probability of rolling a number greater than five on an eight-sided die -/
def prob_second_die : ℚ := 3 / 8

/-- The probability of both events occurring -/
def prob_both : ℚ := prob_first_die * prob_second_die

theorem dice_roll_probability : prob_both = 3 / 16 := by
  sorry

end dice_roll_probability_l1500_150061


namespace quadratic_inequality_implies_range_l1500_150003

theorem quadratic_inequality_implies_range (x : ℝ) : 
  x^2 - 7*x + 12 < 0 → 42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end quadratic_inequality_implies_range_l1500_150003


namespace divisibility_problem_l1500_150087

theorem divisibility_problem (n m k : ℕ) (h1 : n = 172835) (h2 : m = 136) (h3 : k = 21) :
  (n + k) % m = 0 := by
  sorry

end divisibility_problem_l1500_150087


namespace probability_two_non_defective_pens_l1500_150022

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 10) (h2 : defective_pens = 3) :
  let non_defective := total_pens - defective_pens
  let prob_first := non_defective / total_pens
  let prob_second := (non_defective - 1) / (total_pens - 1)
  prob_first * prob_second = 7 / 15 := by sorry

end probability_two_non_defective_pens_l1500_150022


namespace fish_ratio_l1500_150063

/-- The number of fish in Billy's aquarium -/
def billy_fish : ℕ := 10

/-- The number of fish in Tony's aquarium -/
def tony_fish : ℕ := billy_fish * 3

/-- The number of fish in Sarah's aquarium -/
def sarah_fish : ℕ := tony_fish + 5

/-- The number of fish in Bobby's aquarium -/
def bobby_fish : ℕ := sarah_fish * 2

/-- The total number of fish in all aquariums -/
def total_fish : ℕ := 145

theorem fish_ratio : 
  billy_fish = 10 ∧ 
  tony_fish = billy_fish * 3 ∧ 
  sarah_fish = tony_fish + 5 ∧ 
  bobby_fish = sarah_fish * 2 ∧ 
  bobby_fish + sarah_fish + tony_fish + billy_fish = total_fish → 
  tony_fish / billy_fish = 3 := by
  sorry

end fish_ratio_l1500_150063


namespace quadratic_function_property_l1500_150035

theorem quadratic_function_property (a m : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end quadratic_function_property_l1500_150035


namespace expected_composite_count_l1500_150092

/-- The number of elements in the set {1, 2, 3, ..., 100} -/
def setSize : ℕ := 100

/-- The number of composite numbers in the set {1, 2, 3, ..., 100} -/
def compositeCount : ℕ := 74

/-- The number of selections made -/
def selectionCount : ℕ := 5

/-- The probability of selecting a composite number -/
def compositeProbability : ℚ := compositeCount / setSize

/-- Expected number of composite numbers when selecting 5 numbers with replacement from {1, 2, 3, ..., 100} -/
theorem expected_composite_count : 
  (selectionCount : ℚ) * compositeProbability = 37 / 10 := by sorry

end expected_composite_count_l1500_150092


namespace number_percentage_problem_l1500_150059

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → (40/100 : ℝ) * N = 420 := by
  sorry

end number_percentage_problem_l1500_150059


namespace mixed_doubles_pairings_l1500_150038

theorem mixed_doubles_pairings (n_men : Nat) (n_women : Nat) : 
  n_men = 5 → n_women = 4 → (n_men.choose 2) * (n_women.choose 2) * 2 = 120 := by
  sorry

end mixed_doubles_pairings_l1500_150038


namespace slope_product_is_two_l1500_150032

/-- Given two lines with slopes m and n, where one line makes twice the angle
    with the horizontal as the other, has 4 times the slope, and is not horizontal,
    prove that the product of their slopes is 2. -/
theorem slope_product_is_two (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 2 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L₁ makes twice the angle
  m = 4 * n →                                                      -- L₁ has 4 times the slope
  m ≠ 0 →                                                          -- L₁ is not horizontal
  m * n = 2 := by
  sorry

end slope_product_is_two_l1500_150032


namespace original_price_calculation_l1500_150083

/-- Calculates the original price of an article given the profit percentage and profit amount. -/
def calculate_original_price (profit_percentage : ℚ) (profit_amount : ℚ) : ℚ :=
  profit_amount / (profit_percentage / 100)

/-- Theorem: Given an article sold at a 50% profit, where the profit is Rs. 750, 
    the original price of the article was Rs. 1500. -/
theorem original_price_calculation :
  calculate_original_price 50 750 = 1500 := by
  sorry

end original_price_calculation_l1500_150083


namespace total_raisins_l1500_150006

def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

theorem total_raisins : yellow_raisins + black_raisins = 0.7 := by
  sorry

end total_raisins_l1500_150006


namespace integer_sum_proof_l1500_150094

theorem integer_sum_proof (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end integer_sum_proof_l1500_150094


namespace smallest_power_of_1512_l1500_150065

theorem smallest_power_of_1512 :
  ∃ (n : ℕ), 1512 * 49 = n^3 ∧
  ∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (m : ℕ), ∃ (k : ℕ), 1512 * x = m^k := by
  sorry

end smallest_power_of_1512_l1500_150065


namespace solution_set_when_a_is_one_minimum_a_for_f_geq_two_l1500_150069

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |x + 2|

-- Part I
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 :=
sorry

-- Part II
theorem minimum_a_for_f_geq_two :
  (∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x ≥ 2) ∧
  (∀ ε > 0, ∃ a x : ℝ, 0 < a ∧ a < 1/2 + ε ∧ f a x < 2) :=
sorry

end solution_set_when_a_is_one_minimum_a_for_f_geq_two_l1500_150069


namespace product_mod_seven_l1500_150028

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 3 := by
  sorry

end product_mod_seven_l1500_150028


namespace clock_distance_theorem_l1500_150056

/-- Represents a clock on the table -/
structure Clock where
  center : ℝ × ℝ
  radius : ℝ

/-- The state of all clocks at a given time -/
def ClockState := List Clock

/-- Calculate the position of the minute hand at a given time -/
def minuteHandPosition (clock : Clock) (time : ℝ) : ℝ × ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to clock centers -/
def sumDistancesToCenters (clocks : ClockState) : ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to minute hand ends at a given time -/
def sumDistancesToMinuteHands (clocks : ClockState) (time : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem clock_distance_theorem (clocks : ClockState) (h : clocks.length = 50) :
  ∃ t : ℝ, sumDistancesToMinuteHands clocks t > sumDistancesToCenters clocks :=
sorry

end clock_distance_theorem_l1500_150056


namespace maria_fair_spending_l1500_150096

def fair_spending (initial_amount spent_on_rides discounted_ride_cost discount_percent
                   borrowed won food_cost found_money lent_money final_amount : ℚ) : Prop :=
  let discounted_ride_spending := discounted_ride_cost * (1 - discount_percent / 100)
  let net_amount := initial_amount - spent_on_rides - discounted_ride_spending + borrowed + won - food_cost + found_money - lent_money
  net_amount - final_amount = 41

theorem maria_fair_spending :
  fair_spending 87 25 4 25 15 10 12 5 20 16 := by sorry

end maria_fair_spending_l1500_150096


namespace boom_boom_language_size_l1500_150012

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The length of a word -/
def word_length : ℕ := 6

/-- The number of words with at least two identical letters -/
def num_words_with_repeats : ℕ := alphabet_size ^ word_length - Nat.factorial alphabet_size

theorem boom_boom_language_size :
  num_words_with_repeats = 45936 :=
sorry

end boom_boom_language_size_l1500_150012


namespace ellipse_major_axis_length_l1500_150015

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisRatio)

/-- Theorem: The length of the major axis is 7 for the given conditions -/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.75 = 7 := by
  sorry

end ellipse_major_axis_length_l1500_150015


namespace greatest_number_with_conditions_l1500_150067

theorem greatest_number_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 150 → (∃ k : ℕ, m = k^2) → m % 3 = 0 → m ≤ n :=
by sorry

end greatest_number_with_conditions_l1500_150067


namespace alex_age_l1500_150088

theorem alex_age (alex_age precy_age : ℕ) : 
  (alex_age + 3 = 3 * (precy_age + 3)) →
  (alex_age - 1 = 7 * (precy_age - 1)) →
  alex_age = 15 := by
sorry

end alex_age_l1500_150088


namespace line_properties_l1500_150070

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the slope angle of a line in degrees --/
def slope_angle (l : Line) : ℝ :=
  sorry

/-- Calculates the y-intercept of a line --/
def y_intercept (l : Line) : ℝ :=
  sorry

/-- The line x + y + 1 = 0 --/
def line : Line :=
  { a := 1, b := 1, c := 1 }

theorem line_properties :
  slope_angle line = 135 ∧ y_intercept line = -1 := by
  sorry

end line_properties_l1500_150070


namespace simplify_expression_l1500_150017

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  Real.sqrt ((a - Real.pi) ^ 2) + |a - 2| = Real.pi - 2 := by
  sorry

end simplify_expression_l1500_150017


namespace seventh_term_is_64_l1500_150066

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end seventh_term_is_64_l1500_150066


namespace no_unique_five_day_august_l1500_150055

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def countDaysInMonth (m : Month) : DayOfWeek → Nat :=
  sorry

/-- July has five Tuesdays and 30 days -/
def july : Month :=
  { days := 30,
    firstDay := sorry }

/-- August follows July and has 30 days -/
def august : Month :=
  { days := 30,
    firstDay := sorry }

/-- There is no unique day that occurs five times in August -/
theorem no_unique_five_day_august :
  ¬ ∃! (d : DayOfWeek), countDaysInMonth august d = 5 :=
sorry

end no_unique_five_day_august_l1500_150055


namespace fermat_prime_condition_l1500_150030

theorem fermat_prime_condition (n : ℕ) :
  Nat.Prime (2^n + 1) → (n = 0 ∨ ∃ α : ℕ, n = 2^α) :=
by sorry

end fermat_prime_condition_l1500_150030


namespace expected_blue_correct_without_replacement_more_reliable_l1500_150076

-- Define the population
def total_population : ℕ := 200
def blue_population : ℕ := 120
def pink_population : ℕ := 80

-- Define the sample sizes
def sample_size_small : ℕ := 2
def sample_size_large : ℕ := 10

-- Define the true proportion of blue items
def true_proportion : ℚ := blue_population / total_population

-- Part 1: Expected number of blue items in small sample
def expected_blue_small_sample : ℚ := 6/5

-- Part 2: Probabilities for large sample
def prob_within_error_with_replacement : ℚ := 66647/100000
def prob_within_error_without_replacement : ℚ := 67908/100000

-- Theorem statements
theorem expected_blue_correct : 
  ∀ (sampling_method : String), 
  (sampling_method = "with_replacement" ∨ sampling_method = "without_replacement") → 
  expected_blue_small_sample = sample_size_small * true_proportion :=
sorry

theorem without_replacement_more_reliable :
  prob_within_error_without_replacement > prob_within_error_with_replacement :=
sorry

end expected_blue_correct_without_replacement_more_reliable_l1500_150076


namespace john_newspaper_profit_l1500_150086

/-- Calculates the profit made by John selling newspapers --/
theorem john_newspaper_profit :
  let total_newspapers : ℕ := 500
  let selling_price : ℚ := 2
  let sold_percentage : ℚ := 80 / 100
  let discount_percentage : ℚ := 75 / 100
  let profit : ℚ := (total_newspapers : ℚ) * sold_percentage * selling_price - 
                    total_newspapers * (selling_price * (1 - discount_percentage))
  profit = 550
  := by sorry

end john_newspaper_profit_l1500_150086


namespace valuable_files_after_three_rounds_l1500_150036

def first_round_files : ℕ := 1200
def first_round_delete_percent : ℚ := 80 / 100
def second_round_files : ℕ := 600
def second_round_irrelevant_fraction : ℚ := 4 / 5
def final_round_files : ℕ := 700
def final_round_not_pertinent_percent : ℚ := 65 / 100

theorem valuable_files_after_three_rounds :
  let first_round_valuable := first_round_files - (first_round_files * first_round_delete_percent).floor
  let second_round_valuable := second_round_files - (second_round_files * second_round_irrelevant_fraction).floor
  let final_round_valuable := final_round_files - (final_round_files * final_round_not_pertinent_percent).floor
  first_round_valuable + second_round_valuable + final_round_valuable = 605 := by
sorry


end valuable_files_after_three_rounds_l1500_150036


namespace husband_towel_usage_l1500_150052

/-- The number of bath towels used by Kylie in a month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels used by Kylie's daughters in a month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The number of loads of laundry needed to clean all used towels -/
def loads_of_laundry : ℕ := 3

/-- The number of bath towels used by the husband in a month -/
def husband_towels : ℕ := 3

theorem husband_towel_usage :
  kylie_towels + daughters_towels + husband_towels = towels_per_load * loads_of_laundry :=
by sorry

end husband_towel_usage_l1500_150052


namespace even_function_solution_set_l1500_150033

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin -/
def HasSymmetricDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Set.Icc (1 + a) 1 = Set.Icc (-1) 1

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : f = fun x ↦ a * x^2 + b * x + 2)
  (h2 : IsEven f)
  (h3 : HasSymmetricDomain f a) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := by
  sorry

end even_function_solution_set_l1500_150033


namespace m_range_l1500_150034

/-- The function g(x) = mx + 2 -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The closed interval [-1, 2] -/
def I : Set ℝ := Set.Icc (-1) 2

theorem m_range :
  (∀ m : ℝ, (∀ x₁ ∈ I, ∃ x₀ ∈ I, g m x₁ = f x₀) ↔ m ∈ Set.Icc (-1) (1/2)) :=
sorry

end m_range_l1500_150034


namespace tom_video_game_spending_l1500_150026

/-- The amount Tom spent on the Batman game -/
def batman_game_cost : ℚ := 13.6

/-- The amount Tom spent on the Superman game -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_game_cost + superman_game_cost

theorem tom_video_game_spending :
  total_spent = 18.66 := by sorry

end tom_video_game_spending_l1500_150026


namespace ellipse_equation_l1500_150080

/-- The standard equation of an ellipse passing through (3,0) with eccentricity √6/3 -/
theorem ellipse_equation (x y : ℝ) :
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : ℝ × ℝ := (3, 0)
  (∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧
                 (3^2 / a^2 + 0^2 / b^2 = 1) ∧
                 e^2 = 1 - (min a b)^2 / (max a b)^2)) →
  (x^2 / 9 + y^2 / 3 = 1) ∨ (x^2 / 9 + y^2 / 27 = 1) :=
by sorry

end ellipse_equation_l1500_150080


namespace cubic_equation_roots_l1500_150024

theorem cubic_equation_roots :
  ∃ (pos_roots : ℕ), 
    (pos_roots = 1 ∨ pos_roots = 3) ∧
    (∀ x : ℝ, x^3 - 3*x^2 + 4*x - 12 = 0 → x > 0) ∧
    (¬∃ x : ℝ, x < 0 ∧ x^3 - 3*x^2 + 4*x - 12 = 0) := by
  sorry

end cubic_equation_roots_l1500_150024


namespace dandelion_seeds_l1500_150014

theorem dandelion_seeds (S : ℕ) : 
  (2/3 : ℚ) * (5/6 : ℚ) * (1/2 : ℚ) * S = 75 → S = 540 := by
  sorry

end dandelion_seeds_l1500_150014


namespace divisibility_and_expression_l1500_150040

theorem divisibility_and_expression (k : ℕ) : 
  30^k ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end divisibility_and_expression_l1500_150040


namespace cube_greater_than_l1500_150053

theorem cube_greater_than (x y : ℝ) (h : x > y) : x^3 > y^3 := by
  sorry

end cube_greater_than_l1500_150053


namespace white_balls_in_box_l1500_150025

theorem white_balls_in_box (orange_balls black_balls : ℕ) 
  (prob_not_orange_or_white : ℚ) (white_balls : ℕ) : 
  orange_balls = 8 → 
  black_balls = 7 → 
  prob_not_orange_or_white = 38095238095238093 / 100000000000000000 →
  (black_balls : ℚ) / (orange_balls + black_balls + white_balls : ℚ) = prob_not_orange_or_white →
  white_balls = 3 := by
sorry

end white_balls_in_box_l1500_150025


namespace atomic_number_calculation_l1500_150064

/-- Represents an atomic element -/
structure Element where
  massNumber : ℕ
  neutronCount : ℕ
  atomicNumber : ℕ

/-- The relation between mass number, neutron count, and atomic number in an element -/
def isValidElement (e : Element) : Prop :=
  e.massNumber = e.neutronCount + e.atomicNumber

theorem atomic_number_calculation (e : Element)
  (h1 : e.massNumber = 288)
  (h2 : e.neutronCount = 169)
  (h3 : isValidElement e) :
  e.atomicNumber = 119 := by
  sorry

#check atomic_number_calculation

end atomic_number_calculation_l1500_150064


namespace train_speed_l1500_150091

/-- The speed of a train given the time to cross an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 12 →
  platform_length = 320 →
  platform_time = 44 →
  ∃ (train_length : ℝ) (speed_mps : ℝ),
    train_length = speed_mps * pole_time ∧
    train_length + platform_length = speed_mps * platform_time ∧
    speed_mps * 3.6 = 36 := by
  sorry

#check train_speed

end train_speed_l1500_150091


namespace lawrence_walk_l1500_150051

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that given a speed of 3 km/h and a time of 1.33 hours, 
    the distance traveled is 3.99 km -/
theorem lawrence_walk : distance 3 1.33 = 3.99 := by
  sorry

end lawrence_walk_l1500_150051


namespace binomial_18_6_l1500_150043

theorem binomial_18_6 : (Nat.choose 18 6) = 18564 := by
  sorry

end binomial_18_6_l1500_150043


namespace elementary_to_kindergarten_ratio_is_two_to_one_l1500_150031

/-- Represents the purchase of dinosaur models by a school --/
structure ModelPurchase where
  regular_price : ℕ  -- Regular price of each model in dollars
  kindergarten_models : ℕ  -- Number of models for kindergarten library
  total_paid : ℕ  -- Total amount paid in dollars
  discount_percent : ℕ  -- Discount percentage applied

/-- Calculates the ratio of elementary library models to kindergarten library models --/
def elementary_to_kindergarten_ratio (purchase : ModelPurchase) : ℚ :=
  let discounted_price := purchase.regular_price * (100 - purchase.discount_percent) / 100
  let kindergarten_cost := purchase.kindergarten_models * purchase.regular_price
  let elementary_cost := purchase.total_paid - kindergarten_cost
  let elementary_models := elementary_cost / discounted_price
  elementary_models / purchase.kindergarten_models

/-- Theorem stating the ratio of elementary to kindergarten models is 2:1 --/
theorem elementary_to_kindergarten_ratio_is_two_to_one 
  (purchase : ModelPurchase)
  (h1 : purchase.regular_price = 100)
  (h2 : purchase.kindergarten_models = 2)
  (h3 : purchase.total_paid = 570)
  (h4 : purchase.discount_percent = 5)
  (h5 : purchase.kindergarten_models + 
        (purchase.total_paid - purchase.kindergarten_models * purchase.regular_price) / 
        (purchase.regular_price * (100 - purchase.discount_percent) / 100) > 5) :
  elementary_to_kindergarten_ratio purchase = 2 := by
  sorry

end elementary_to_kindergarten_ratio_is_two_to_one_l1500_150031
