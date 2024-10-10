import Mathlib

namespace opposite_black_is_orange_l89_8998

-- Define the colors
inductive Color
| Orange | Yellow | Blue | Pink | Violet | Black

-- Define a cube face
structure Face :=
  (color : Color)

-- Define a cube
structure Cube :=
  (top : Face)
  (front : Face)
  (right : Face)
  (bottom : Face)
  (back : Face)
  (left : Face)

-- Define the views
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Pink

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Violet ∧ c.right.color = Color.Pink

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Orange ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Pink

-- Theorem statement
theorem opposite_black_is_orange (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Black →
  c.top.color = Color.Orange :=
sorry

end opposite_black_is_orange_l89_8998


namespace firewood_collection_l89_8923

/-- Firewood collection problem -/
theorem firewood_collection (K H : ℝ) (x E : ℝ) 
  (hK : K = 0.8)
  (hH : H = 1.5)
  (eq1 : 10 * K + x * E + 12 * H = 44)
  (eq2 : 10 + x + 12 = 35) :
  x = 13 ∧ E = 18 / 13 := by
  sorry

end firewood_collection_l89_8923


namespace product_approximation_l89_8958

def is_approximately_equal (x y : ℕ) (tolerance : ℕ) : Prop :=
  (x ≤ y + tolerance) ∧ (y ≤ x + tolerance)

theorem product_approximation (tolerance : ℕ) :
  (is_approximately_equal (4 * 896) 3600 tolerance) ∧
  (is_approximately_equal (405 * 9) 3600 tolerance) ∧
  ¬(is_approximately_equal (6 * 689) 3600 tolerance) ∧
  ¬(is_approximately_equal (398 * 8) 3600 tolerance) :=
by sorry

end product_approximation_l89_8958


namespace complex_inverse_calculation_l89_8978

theorem complex_inverse_calculation (i : ℂ) (h : i^2 = -1) : 
  (2*i - 3*i⁻¹)⁻¹ = -i/5 := by sorry

end complex_inverse_calculation_l89_8978


namespace structure_has_112_cubes_l89_8947

/-- A structure made of cubes with 5 layers -/
structure CubeStructure where
  middle_layer : ℕ
  other_layers : ℕ
  total_layers : ℕ
  h_middle : middle_layer = 16
  h_other : other_layers = 24
  h_total : total_layers = 5

/-- The total number of cubes in the structure -/
def total_cubes (s : CubeStructure) : ℕ :=
  s.middle_layer + (s.total_layers - 1) * s.other_layers

/-- Theorem stating that the structure contains 112 cubes -/
theorem structure_has_112_cubes (s : CubeStructure) : total_cubes s = 112 := by
  sorry


end structure_has_112_cubes_l89_8947


namespace potato_percentage_l89_8932

/-- Proves that the percentage of cleared land planted with potato is 30% -/
theorem potato_percentage (total_land : ℝ) (cleared_land : ℝ) (grape_land : ℝ) (tomato_land : ℝ) 
  (h1 : total_land = 3999.9999999999995)
  (h2 : cleared_land = 0.9 * total_land)
  (h3 : grape_land = 0.6 * cleared_land)
  (h4 : tomato_land = 360)
  : (cleared_land - grape_land - tomato_land) / cleared_land = 0.3 := by
  sorry

#eval (3999.9999999999995 * 0.9 - 3999.9999999999995 * 0.9 * 0.6 - 360) / (3999.9999999999995 * 0.9)

end potato_percentage_l89_8932


namespace no_primes_in_sequence_l89_8972

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: The sequence is increasing -/
def IsIncreasing (s : Sequence) : Prop :=
  ∀ n : ℕ, s n < s (n + 1)

/-- Property: Any three consecutive numbers form an arithmetic or geometric progression -/
def IsArithmeticOrGeometric (s : Sequence) : Prop :=
  ∀ n : ℕ, (2 * s (n + 1) = s n + s (n + 2)) ∨ (s (n + 1) ^ 2 = s n * s (n + 2))

/-- Property: The first two numbers are divisible by 4 -/
def FirstTwoDivisibleByFour (s : Sequence) : Prop :=
  4 ∣ s 0 ∧ 4 ∣ s 1

/-- Main theorem -/
theorem no_primes_in_sequence (s : Sequence)
  (h_inc : IsIncreasing s)
  (h_prog : IsArithmeticOrGeometric s)
  (h_div4 : FirstTwoDivisibleByFour s) :
  ∀ n : ℕ, ¬ Nat.Prime (s n) :=
sorry

end no_primes_in_sequence_l89_8972


namespace student_calculation_l89_8930

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 48) : 
  chosen_number * 5 - 138 = 102 := by
  sorry

end student_calculation_l89_8930


namespace breakfast_dessert_l89_8993

-- Define the possible breakfast items
inductive BreakfastItem
  | Whiskey
  | Duck
  | Oranges
  | Pie
  | BelleHelenePear
  | StrawberrySherbet
  | Coffee

-- Define the structure of a journalist's statement
structure JournalistStatement where
  items : List BreakfastItem

-- Define the honesty levels of journalists
inductive JournalistHonesty
  | AlwaysTruthful
  | OneFalseStatement
  | AlwaysLies

-- Define the breakfast observation
structure BreakfastObservation where
  jules : JournalistStatement
  jacques : JournalistStatement
  jim : JournalistStatement
  julesHonesty : JournalistHonesty
  jacquesHonesty : JournalistHonesty
  jimHonesty : JournalistHonesty

def breakfast : BreakfastObservation := {
  jules := { items := [BreakfastItem.Whiskey, BreakfastItem.Duck, BreakfastItem.Oranges, BreakfastItem.Coffee] },
  jacques := { items := [BreakfastItem.Pie, BreakfastItem.BelleHelenePear] },
  jim := { items := [BreakfastItem.Whiskey, BreakfastItem.Pie, BreakfastItem.StrawberrySherbet, BreakfastItem.Coffee] },
  julesHonesty := JournalistHonesty.AlwaysTruthful,
  jacquesHonesty := JournalistHonesty.AlwaysLies,
  jimHonesty := JournalistHonesty.OneFalseStatement
}

theorem breakfast_dessert :
  ∃ (dessert : BreakfastItem), dessert = BreakfastItem.StrawberrySherbet :=
by sorry

end breakfast_dessert_l89_8993


namespace isosceles_triangle_perimeter_is_7_or_8_l89_8938

def isosceles_triangle_perimeter (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧  -- positive side lengths
  (x = y ∨ x + y > x)  -- triangle inequality
  ∧ y = Real.sqrt (2 - x) + Real.sqrt (3 * x - 6) + 3

theorem isosceles_triangle_perimeter_is_7_or_8 :
  ∀ x y : ℝ, isosceles_triangle_perimeter x y →
  (x + y + (if x = y then x else y) = 7 ∨ x + y + (if x = y then x else y) = 8) :=
sorry

end isosceles_triangle_perimeter_is_7_or_8_l89_8938


namespace eggs_left_after_recovering_capital_l89_8964

theorem eggs_left_after_recovering_capital 
  (total_eggs : ℕ) 
  (crate_cost_cents : ℕ) 
  (selling_price_cents : ℕ) : ℕ :=
  let eggs_sold := crate_cost_cents / selling_price_cents
  total_eggs - eggs_sold

#check eggs_left_after_recovering_capital 30 500 20 = 5

end eggs_left_after_recovering_capital_l89_8964


namespace quadratic_two_distinct_roots_l89_8981

/-- A quadratic equation kx^2 - 4x + 1 = 0 has two distinct real roots if and only if k < 4 and k ≠ 0 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 4 * x + 1 = 0 ∧ k * y^2 - 4 * y + 1 = 0) ↔ 
  (k < 4 ∧ k ≠ 0) :=
sorry

end quadratic_two_distinct_roots_l89_8981


namespace plane_equation_theorem_l89_8949

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The foot of the perpendicular from the origin to the plane -/
def footOfPerpendicular : Point3D :=
  { x := 10, y := -5, z := 4 }

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (coeff : PlaneCoefficients) : Prop :=
  coeff.A > 0 ∧ Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.B) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.C) = 1 ∧
  Nat.gcd (Int.natAbs coeff.A) (Int.natAbs coeff.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesPlaneEquation (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The main theorem to prove -/
theorem plane_equation_theorem :
  ∃ (coeff : PlaneCoefficients),
    validCoefficients coeff ∧
    satisfiesPlaneEquation footOfPerpendicular coeff ∧
    coeff.A = 10 ∧ coeff.B = -5 ∧ coeff.C = 4 ∧ coeff.D = -141 :=
sorry

end plane_equation_theorem_l89_8949


namespace no_real_roots_l89_8984

theorem no_real_roots : ∀ x : ℝ, x^2 - x * Real.sqrt 5 + Real.sqrt 2 ≠ 0 := by
  sorry

end no_real_roots_l89_8984


namespace evaporation_weight_theorem_l89_8954

/-- Represents the weight of a glass containing a solution --/
structure GlassSolution where
  total_weight : ℝ
  water_percentage : ℝ
  glass_weight : ℝ

/-- Calculates the final weight of a glass solution after water evaporation --/
def final_weight (initial : GlassSolution) (final_water_percentage : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the initial conditions and final water percentage,
    the final weight of the glass with solution is 400 grams --/
theorem evaporation_weight_theorem (initial : GlassSolution) 
    (h1 : initial.total_weight = 500)
    (h2 : initial.water_percentage = 0.99)
    (h3 : initial.glass_weight = 300)
    (final_water_percentage : ℝ)
    (h4 : final_water_percentage = 0.98) :
    final_weight initial final_water_percentage = 400 := by
  sorry

end evaporation_weight_theorem_l89_8954


namespace arithmetic_sequence_length_l89_8901

theorem arithmetic_sequence_length : 
  ∀ (a₁ : ℝ) (d : ℝ) (aₙ : ℝ),
  a₁ = 2.5 → d = 5 → aₙ = 72.5 →
  ∃ (n : ℕ), n = 15 ∧ aₙ = a₁ + (n - 1) * d :=
by sorry

end arithmetic_sequence_length_l89_8901


namespace no_20_digit_square_starting_with_11_ones_l89_8915

theorem no_20_digit_square_starting_with_11_ones :
  ¬∃ (n : ℕ), 
    (10^19 ≤ n) ∧ 
    (n < 10^20) ∧ 
    (11111111111 * 10^9 ≤ n) ∧ 
    (n < 11111111112 * 10^9) ∧ 
    (∃ (k : ℕ), n = k^2) :=
by sorry

end no_20_digit_square_starting_with_11_ones_l89_8915


namespace sock_pair_count_l89_8977

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from a drawer containing 5 white socks, 5 brown socks, 3 blue socks,
    and 2 red socks is equal to 81. -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by
  sorry

end sock_pair_count_l89_8977


namespace log_difference_negative_l89_8962

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end log_difference_negative_l89_8962


namespace sandy_marks_l89_8975

theorem sandy_marks (correct_marks : ℕ) (incorrect_marks : ℕ) (total_sums : ℕ) (correct_sums : ℕ) 
  (h1 : correct_marks = 3)
  (h2 : incorrect_marks = 2)
  (h3 : total_sums = 30)
  (h4 : correct_sums = 22) :
  correct_marks * correct_sums - incorrect_marks * (total_sums - correct_sums) = 50 := by
  sorry

end sandy_marks_l89_8975


namespace down_payment_correct_l89_8970

/-- Represents the down payment problem for a car purchase. -/
structure DownPayment where
  total : ℕ
  contributionA : ℕ
  contributionB : ℕ
  contributionC : ℕ
  contributionD : ℕ

/-- Theorem stating that the given contributions satisfy the problem conditions. -/
theorem down_payment_correct (dp : DownPayment) : 
  dp.total = 3500 ∧
  dp.contributionA = 1225 ∧
  dp.contributionB = 875 ∧
  dp.contributionC = 700 ∧
  dp.contributionD = 700 ∧
  dp.contributionA + dp.contributionB + dp.contributionC + dp.contributionD = dp.total ∧
  dp.contributionA = (35 * dp.total) / 100 ∧
  dp.contributionB = (25 * dp.total) / 100 ∧
  dp.contributionC = (20 * dp.total) / 100 ∧
  dp.contributionD = dp.total - (dp.contributionA + dp.contributionB + dp.contributionC) :=
by sorry


end down_payment_correct_l89_8970


namespace zoo_animal_ratio_l89_8939

/-- Prove the ratio of monkeys to camels at the zoo -/
theorem zoo_animal_ratio : 
  ∀ (zebras camels monkeys giraffes : ℕ),
    zebras = 12 →
    camels = zebras / 2 →
    ∃ k : ℕ, monkeys = k * camels →
    giraffes = 2 →
    monkeys = giraffes + 22 →
    monkeys / camels = 4 := by
  sorry

end zoo_animal_ratio_l89_8939


namespace problem_proof_l89_8948

theorem problem_proof (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 := by
  sorry

end problem_proof_l89_8948


namespace fraction_equality_l89_8991

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c^2 / d = 16) :
  d / a = 1 / 25 := by
  sorry

end fraction_equality_l89_8991


namespace arithmetic_sequence_difference_l89_8980

def is_arithmetic_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ c - b = r ∧ d - c = r ∧ e - d = r

theorem arithmetic_sequence_difference 
  (a b c : ℝ) (h : is_arithmetic_sequence 2 a b c 9) : 
  c - a = (7 : ℝ) / 2 := by
  sorry

end arithmetic_sequence_difference_l89_8980


namespace inverse_sum_zero_l89_8942

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 1; 7, 3]

theorem inverse_sum_zero :
  ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), 
    A * B = 1 ∧ B * A = 1 →
    B 0 0 + B 0 1 + B 1 0 + B 1 1 = 0 := by
  sorry

end inverse_sum_zero_l89_8942


namespace power_of_five_cube_l89_8951

theorem power_of_five_cube (n : ℤ) : 
  (∃ a : ℕ, n^3 - 3*n^2 + n + 2 = 5^a) ↔ n = 3 :=
by sorry

end power_of_five_cube_l89_8951


namespace probability_at_least_one_type_b_l89_8979

def total_questions : ℕ := 5
def type_a_questions : ℕ := 2
def type_b_questions : ℕ := 3
def selected_questions : ℕ := 2

theorem probability_at_least_one_type_b :
  let total_combinations := Nat.choose total_questions selected_questions
  let all_type_a_combinations := Nat.choose type_a_questions selected_questions
  (total_combinations - all_type_a_combinations) / total_combinations = 9 / 10 := by
  sorry

end probability_at_least_one_type_b_l89_8979


namespace parabola_vertex_l89_8988

/-- The vertex of the parabola y = 2x^2 + 16x + 34 is (-4, 2) -/
theorem parabola_vertex :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 34
  ∃! (h k : ℝ), ∀ x, f x = 2 * (x - h)^2 + k ∧ h = -4 ∧ k = 2 :=
by sorry

end parabola_vertex_l89_8988


namespace consecutive_odd_integers_sum_l89_8907

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ n : ℤ, a = n ∧ 
    (∀ i : Fin 5, Odd (a + 2 * i.val)) ∧
    (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = -365)) → 
  (a + 8 = -69) := by
  sorry

end consecutive_odd_integers_sum_l89_8907


namespace division_of_decimals_l89_8959

theorem division_of_decimals : (2.4 : ℝ) / 0.06 = 40 := by sorry

end division_of_decimals_l89_8959


namespace alok_mixed_veg_plates_l89_8968

/-- Represents the order and pricing information for a restaurant bill --/
structure RestaurantBill where
  chapatis : ℕ
  rice : ℕ
  iceCream : ℕ
  chapatiPrice : ℕ
  ricePrice : ℕ
  mixedVegPrice : ℕ
  iceCreamPrice : ℕ
  totalPaid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixedVegPlates (bill : RestaurantBill) : ℕ :=
  (bill.totalPaid - (bill.chapatis * bill.chapatiPrice + bill.rice * bill.ricePrice + bill.iceCream * bill.iceCreamPrice)) / bill.mixedVegPrice

/-- Theorem stating that Alok ordered 7 plates of mixed vegetable --/
theorem alok_mixed_veg_plates :
  let bill : RestaurantBill := {
    chapatis := 16,
    rice := 5,
    iceCream := 6,
    chapatiPrice := 6,
    ricePrice := 45,
    mixedVegPrice := 70,
    iceCreamPrice := 40,
    totalPaid := 1051
  }
  mixedVegPlates bill = 7 := by
  sorry

end alok_mixed_veg_plates_l89_8968


namespace ant_meeting_point_l89_8929

/-- Triangle with given side lengths --/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point on the perimeter of the triangle --/
structure PerimeterPoint where
  side : Fin 3
  distance : ℝ

/-- Represents the meeting point of two ants --/
def MeetingPoint (t : Triangle) (w : PerimeterPoint) : Prop :=
  w.side = 1 ∧ w.distance ≤ t.yz

/-- The distance YW --/
def YW (t : Triangle) (w : PerimeterPoint) : ℝ :=
  t.yz - w.distance

/-- Main theorem --/
theorem ant_meeting_point (t : Triangle) (w : PerimeterPoint) :
  t.xy = 8 ∧ t.yz = 10 ∧ t.xz = 12 ∧ MeetingPoint t w →
  YW t w = 3 := by
  sorry

end ant_meeting_point_l89_8929


namespace calculation_proofs_l89_8955

theorem calculation_proofs :
  (1 - 2^3 / 8 - 1/4 * (-2)^2 = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end calculation_proofs_l89_8955


namespace reciprocal_sum_theorem_l89_8916

theorem reciprocal_sum_theorem (a b c : ℝ) (h : 1 / a + 1 / b = 1 / c) : c = a * b / (b + a) := by
  sorry

end reciprocal_sum_theorem_l89_8916


namespace coach_b_baseballs_l89_8961

/-- The number of baseballs Coach B bought -/
def num_baseballs : ℕ := 14

/-- The cost of each basketball -/
def basketball_cost : ℚ := 29

/-- The cost of each baseball -/
def baseball_cost : ℚ := 5/2

/-- The cost of the baseball bat -/
def bat_cost : ℚ := 18

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The difference in spending between Coach A and Coach B -/
def spending_difference : ℚ := 237

theorem coach_b_baseballs :
  (num_basketballs * basketball_cost) = 
  spending_difference + (num_baseballs * baseball_cost + bat_cost) :=
by sorry

end coach_b_baseballs_l89_8961


namespace special_ellipse_eccentricity_l89_8987

/-- An ellipse with the property that the minimum distance from a point on the ellipse to a directrix is equal to the semi-latus rectum. -/
structure SpecialEllipse where
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ
  /-- The semi-latus rectum of the ellipse -/
  semiLatusRectum : ℝ
  /-- The minimum distance from a point on the ellipse to a directrix -/
  minDirectrixDistance : ℝ
  /-- The condition that the minimum distance to a directrix equals the semi-latus rectum -/
  distance_eq_semiLatusRectum : minDirectrixDistance = semiLatusRectum

/-- The eccentricity of a special ellipse is √2/2 -/
theorem special_ellipse_eccentricity (e : SpecialEllipse) : e.eccentricity = Real.sqrt 2 / 2 :=
  sorry

end special_ellipse_eccentricity_l89_8987


namespace compute_expression_l89_8957

theorem compute_expression : 8 * (243 / 3 + 81 / 9 + 25 / 25 + 3) = 752 := by
  sorry

end compute_expression_l89_8957


namespace smallest_digits_to_append_l89_8989

def is_divisible_by_all_less_than_10 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10 → k > 0 → n % k = 0

def append_digits (base n digits : ℕ) : ℕ :=
  base * (10 ^ digits) + n

theorem smallest_digits_to_append : 
  (∀ k < 4, ¬∃ n : ℕ, n < 10^k ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n k)) ∧
  (∃ n : ℕ, n < 10^4 ∧ is_divisible_by_all_less_than_10 (append_digits 2014 n 4)) :=
sorry

end smallest_digits_to_append_l89_8989


namespace root_in_interval_l89_8945

-- Define the function f(x) = x^3 + 3x - 1
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- State the theorem
theorem root_in_interval :
  (f 0 < 0) → (f 1 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 := by
  sorry

end root_in_interval_l89_8945


namespace f_seven_plus_f_nine_l89_8911

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_seven_plus_f_nine (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 4)
  (h_odd : is_odd (fun x ↦ f (x - 1)))
  (h_f_one : f 1 = 1) : 
  f 7 + f 9 = 1 := by
  sorry

end f_seven_plus_f_nine_l89_8911


namespace log_property_l89_8982

theorem log_property (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) (h4 : f 9 = 2) : 
  f (a ^ a) = 3 := by
sorry

end log_property_l89_8982


namespace two_dice_outcomes_l89_8927

/-- The number of possible outcomes for a single die. -/
def outcomes_per_die : ℕ := 6

/-- The total number of possible outcomes when throwing two identical dice simultaneously. -/
def total_outcomes : ℕ := outcomes_per_die * outcomes_per_die

/-- Theorem stating that the total number of possible outcomes when throwing two identical dice simultaneously is 36. -/
theorem two_dice_outcomes : total_outcomes = 36 := by
  sorry

end two_dice_outcomes_l89_8927


namespace not_divisible_by_1955_l89_8971

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end not_divisible_by_1955_l89_8971


namespace expression_value_l89_8940

theorem expression_value : (64 + 27)^2 - (27^2 + 64^2) + 3 * Real.rpow 1728 (1/3) = 3492 := by
  sorry

end expression_value_l89_8940


namespace bin_game_expected_value_l89_8925

theorem bin_game_expected_value (k : ℕ) (h1 : k > 0) : 
  (8 / (8 + k : ℝ)) * 3 + (k / (8 + k : ℝ)) * (-1) = 1 → k = 8 :=
by sorry

end bin_game_expected_value_l89_8925


namespace log_inequality_l89_8960

theorem log_inequality : ∃ (a b : ℝ), 
  (a = Real.log 0.8 / Real.log 0.7) ∧ 
  (b = Real.log 0.4 / Real.log 0.5) ∧ 
  (b > a) ∧ (a > 0) := by
  sorry

end log_inequality_l89_8960


namespace least_common_denominator_l89_8946

theorem least_common_denominator (a b c d e f g h : ℕ) 
  (ha : a = 2) (hb : b = 3) (hc : c = 4) (hd : d = 5) 
  (he : e = 6) (hf : f = 7) (hg : g = 9) (hh : h = 10) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f (Nat.lcm g h)))))) = 2520 := by
  sorry

end least_common_denominator_l89_8946


namespace hezekiah_age_l89_8912

/-- Given that Ryanne is 7 years older than Hezekiah and their combined age is 15, 
    prove that Hezekiah is 4 years old. -/
theorem hezekiah_age (hezekiah_age ryanne_age : ℕ) 
  (h1 : ryanne_age = hezekiah_age + 7)
  (h2 : hezekiah_age + ryanne_age = 15) : 
  hezekiah_age = 4 := by
  sorry

end hezekiah_age_l89_8912


namespace real_estate_investment_l89_8941

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 6) :
  let mutual_funds := total_investment / (1 + real_estate_ratio)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 171428.58 := by sorry

end real_estate_investment_l89_8941


namespace decreasing_function_l89_8935

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end decreasing_function_l89_8935


namespace figure_side_length_l89_8974

theorem figure_side_length (total_area : ℝ) (y : ℝ) : 
  total_area = 1300 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by sorry

end figure_side_length_l89_8974


namespace rectangle_perimeter_l89_8918

/-- Given a square with side length y that is divided into a central square
    with side length (y - z) and four congruent rectangles, prove that
    the perimeter of one of these rectangles is 2y. -/
theorem rectangle_perimeter (y z : ℝ) (hz : z < y) :
  let central_side := y - z
  let rect_long_side := y - z
  let rect_short_side := y - (y - z)
  2 * rect_long_side + 2 * rect_short_side = 2 * y :=
by sorry

end rectangle_perimeter_l89_8918


namespace compute_expression_l89_8908

theorem compute_expression : (85 * 1515 - 25 * 1515) + (48 * 1515) = 163620 := by
  sorry

end compute_expression_l89_8908


namespace polynomial_expansion_l89_8976

theorem polynomial_expansion (z : ℝ) : 
  (2 * z^2 + 5 * z - 6) * (3 * z^3 - 2 * z + 1) = 
  6 * z^5 + 15 * z^4 - 22 * z^3 - 8 * z^2 + 17 * z - 6 := by
  sorry

end polynomial_expansion_l89_8976


namespace custom_op_neg_four_six_l89_8909

-- Define the custom operation ﹡
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b - 1

-- Theorem statement
theorem custom_op_neg_four_six :
  custom_op (-4) 6 = -9 := by
  sorry

end custom_op_neg_four_six_l89_8909


namespace line_circle_intersection_l89_8914

theorem line_circle_intersection (k : ℝ) : ∃ (x y : ℝ),
  y = k * x - k ∧ (x - 2)^2 + y^2 = 3 := by
  sorry

end line_circle_intersection_l89_8914


namespace complex_equation_solution_l89_8926

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end complex_equation_solution_l89_8926


namespace intersection_of_A_and_B_l89_8910

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end intersection_of_A_and_B_l89_8910


namespace smallest_non_five_divisible_unit_l89_8990

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def divisible_by_five_units (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

theorem smallest_non_five_divisible_unit : 
  (∀ d, is_digit d → (∀ n, divisible_by_five_units n → n % 10 ≠ d) → d ≥ 1) ∧
  (∃ n, divisible_by_five_units n ∧ n % 10 ≠ 1) :=
sorry

end smallest_non_five_divisible_unit_l89_8990


namespace triangle_count_l89_8985

/-- The number of triangles formed by three mutually intersecting line segments
    in a configuration of n points on a circle, where n ≥ 6 and
    any three line segments do not intersect at a single point inside the circle. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating the number of triangles formed under the given conditions. -/
theorem triangle_count (n : ℕ) (h : n ≥ 6) :
  num_triangles n = Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end triangle_count_l89_8985


namespace fuel_cost_calculation_l89_8967

/-- Calculates the total cost to fill up both a truck's diesel tank and a car's gasoline tank --/
def total_fuel_cost (truck_capacity : ℝ) (car_capacity : ℝ) (truck_fullness : ℝ) (car_fullness : ℝ) (diesel_price : ℝ) (gasoline_price : ℝ) : ℝ :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  truck_to_fill * diesel_price + car_to_fill * gasoline_price

/-- Theorem stating that the total cost to fill up both tanks is $75.75 --/
theorem fuel_cost_calculation :
  total_fuel_cost 25 15 0.5 (1/3) 3.5 3.2 = 75.75 := by
  sorry

end fuel_cost_calculation_l89_8967


namespace pell_equation_solutions_l89_8921

theorem pell_equation_solutions :
  let solutions : List (ℤ × ℤ) := [(2, 1), (7, 4), (26, 15), (97, 56)]
  ∀ (x y : ℤ), (x, y) ∈ solutions → x^2 - 3*y^2 = 1 :=
by sorry

end pell_equation_solutions_l89_8921


namespace exists_password_with_twenty_combinations_l89_8903

/-- Represents a character in the password --/
structure PasswordChar :=
  (value : Char)

/-- Represents a 5-character password --/
structure Password :=
  (chars : Fin 5 → PasswordChar)

/-- Counts the number of unique permutations of a password --/
def countUniqueCombinations (password : Password) : ℕ :=
  sorry

/-- Theorem: There exists a 5-character password with exactly 20 different combinations --/
theorem exists_password_with_twenty_combinations : 
  ∃ (password : Password), countUniqueCombinations password = 20 := by
  sorry

end exists_password_with_twenty_combinations_l89_8903


namespace weight_of_new_person_l89_8917

/-- The weight of the new person in a group where the average weight has increased --/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) : ℝ :=
  old_weight + n * avg_increase

/-- Theorem: The weight of the new person in the given scenario is 78.5 kg --/
theorem weight_of_new_person :
  new_person_weight 9 1.5 65 = 78.5 := by
  sorry

end weight_of_new_person_l89_8917


namespace encryption_correspondence_unique_decryption_l89_8969

/-- Encryption function that maps a plaintext to a ciphertext -/
def encrypt (p : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let (a, b, c, d) := p
  (a + 2*b, b + c, 2*c + 3*d, 4*d)

/-- Theorem stating that the plaintext (6, 4, 1, 7) corresponds to the ciphertext (14, 9, 23, 28) -/
theorem encryption_correspondence :
  encrypt (6, 4, 1, 7) = (14, 9, 23, 28) := by
  sorry

/-- Theorem stating that the plaintext (6, 4, 1, 7) is the unique solution -/
theorem unique_decryption :
  ∀ p : ℕ × ℕ × ℕ × ℕ, encrypt p = (14, 9, 23, 28) → p = (6, 4, 1, 7) := by
  sorry

end encryption_correspondence_unique_decryption_l89_8969


namespace unique_two_digit_integer_l89_8937

theorem unique_two_digit_integer (s : ℕ) : 
  (s ≥ 10 ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end unique_two_digit_integer_l89_8937


namespace min_students_required_l89_8956

/-- Represents a set of days in which a student participates -/
def ParticipationSet := Finset (Fin 6)

/-- The property that for any 3 days, there's a student participating in all 3 -/
def CoversAllTriples (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 3 → ∃ s ∈ sets, days ⊆ s

/-- The property that no student participates in all 4 days of any 4-day selection -/
def NoQuadruplesCovered (sets : Finset ParticipationSet) : Prop :=
  ∀ (days : Finset (Fin 6)), days.card = 4 → ∀ s ∈ sets, ¬(days ⊆ s)

/-- The main theorem stating the minimum number of students required -/
theorem min_students_required :
  ∃ (sets : Finset ParticipationSet),
    sets.card = 20 ∧
    (∀ s ∈ sets, s.card = 3) ∧
    CoversAllTriples sets ∧
    NoQuadruplesCovered sets ∧
    (∀ (sets' : Finset ParticipationSet),
      (∀ s' ∈ sets', s'.card = 3) →
      CoversAllTriples sets' →
      NoQuadruplesCovered sets' →
      sets'.card ≥ 20) :=
sorry

end min_students_required_l89_8956


namespace emma_square_calculation_l89_8913

theorem emma_square_calculation : 37^2 = 38^2 - 75 := by
  sorry

end emma_square_calculation_l89_8913


namespace real_roots_of_polynomial_l89_8922

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = -2 ∨ x = 1 := by
sorry

end real_roots_of_polynomial_l89_8922


namespace simple_interest_problem_l89_8943

/-- Given a principal amount P, prove that if the simple interest on P at 4% for 5 years
    is equal to P - 2240, then P = 2800. -/
theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2240 → P = 2800 := by
  sorry

end simple_interest_problem_l89_8943


namespace common_factor_proof_l89_8919

theorem common_factor_proof (a b : ℕ) : 
  (4 * a^2 * b^3).gcd (6 * a^3 * b) = 2 * a^2 * b :=
by sorry

end common_factor_proof_l89_8919


namespace sufficient_not_necessary_condition_l89_8936

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
by sorry

end sufficient_not_necessary_condition_l89_8936


namespace gcd_lcm_problem_l89_8965

theorem gcd_lcm_problem (A B : ℕ) (hA : A = 8 * 6) (hB : B = 36 / 3) :
  Nat.gcd A B = 12 ∧ Nat.lcm A B = 48 := by
  sorry

end gcd_lcm_problem_l89_8965


namespace spherical_coordinate_symmetry_l89_8999

/-- Given a point with rectangular coordinates (3, -4, 2) and corresponding
    spherical coordinates (ρ, θ, φ), prove that the point with spherical
    coordinates (ρ, -θ, φ) has rectangular coordinates (3, -4, 2). -/
theorem spherical_coordinate_symmetry (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = 3 ∧
  ρ * Real.sin φ * Real.sin θ = -4 ∧
  ρ * Real.cos φ = 2 →
  ρ * Real.sin φ * Real.cos (-θ) = 3 ∧
  ρ * Real.sin φ * Real.sin (-θ) = -4 ∧
  ρ * Real.cos φ = 2 :=
by sorry

end spherical_coordinate_symmetry_l89_8999


namespace club_officer_selection_l89_8973

/-- Represents a club with members and officers -/
structure Club where
  totalMembers : ℕ
  officerPositions : ℕ
  aliceAndBob : ℕ

/-- Calculates the number of ways to choose officers in a club -/
def chooseOfficers (club : Club) : ℕ :=
  let remainingMembers := club.totalMembers - club.aliceAndBob
  let case1 := remainingMembers * (remainingMembers - 1) * (remainingMembers - 2) * (remainingMembers - 3)
  let case2 := (club.officerPositions.choose 2) * remainingMembers * (remainingMembers - 1)
  case1 + case2

/-- Theorem stating the number of ways to choose officers in the specific club scenario -/
theorem club_officer_selection :
  let club : Club := { totalMembers := 30, officerPositions := 4, aliceAndBob := 2 }
  chooseOfficers club = 495936 := by
  sorry

end club_officer_selection_l89_8973


namespace co_presidents_selection_l89_8963

theorem co_presidents_selection (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 3) :
  Nat.choose n k = 816 := by
  sorry

end co_presidents_selection_l89_8963


namespace rectangle_perimeter_is_164_l89_8983

/-- Represents the side lengths of the squares in the rectangle dissection -/
structure SquareSides where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  a₅ : ℕ
  a₆ : ℕ
  a₇ : ℕ
  a₈ : ℕ
  a₉ : ℕ

/-- The conditions for the rectangle dissection -/
def RectangleDissectionConditions (s : SquareSides) : Prop :=
  s.a₁ + s.a₂ = s.a₄ ∧
  s.a₁ + s.a₄ = s.a₅ ∧
  s.a₄ + s.a₅ = s.a₇ ∧
  s.a₅ + s.a₇ = s.a₉ ∧
  s.a₂ + s.a₄ + s.a₇ = s.a₈ ∧
  s.a₂ + s.a₈ = s.a₆ ∧
  s.a₁ + s.a₅ + s.a₉ = s.a₃ ∧
  s.a₃ + s.a₆ = s.a₈ + s.a₇

/-- The width of the rectangle -/
def RectangleWidth (s : SquareSides) : ℕ := s.a₄ + s.a₇ + s.a₉

/-- The length of the rectangle -/
def RectangleLength (s : SquareSides) : ℕ := s.a₂ + s.a₈ + s.a₆

/-- The main theorem: Given the conditions, the perimeter of the rectangle is 164 -/
theorem rectangle_perimeter_is_164 (s : SquareSides) 
  (h : RectangleDissectionConditions s) 
  (h_coprime : Nat.Coprime (RectangleWidth s) (RectangleLength s)) :
  2 * (RectangleWidth s + RectangleLength s) = 164 := by
  sorry


end rectangle_perimeter_is_164_l89_8983


namespace only_negative_number_l89_8905

theorem only_negative_number (a b c d : ℝ) : 
  a = |(-2)| ∧ b = Real.sqrt 3 ∧ c = 0 ∧ d = -5 →
  (d < 0 ∧ a ≥ 0 ∧ b > 0 ∧ c = 0) := by sorry

end only_negative_number_l89_8905


namespace ratio_sum_problem_l89_8906

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by sorry

end ratio_sum_problem_l89_8906


namespace trig_inequality_range_l89_8934

theorem trig_inequality_range (x : Real) : 
  (x ∈ Set.Icc 0 Real.pi) → 
  (Real.cos x)^2 > (Real.sin x)^2 → 
  x ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Ioo (3 * Real.pi / 4) Real.pi :=
by sorry

end trig_inequality_range_l89_8934


namespace fifth_power_inequality_l89_8902

theorem fifth_power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + b^3*a*c + c^3*a*b := by
  sorry

end fifth_power_inequality_l89_8902


namespace same_figure_l89_8995

noncomputable section

open Complex

/-- Two equations describe the same figure in the complex plane -/
theorem same_figure (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  {z : ℂ | abs (z + n * I) + abs (z - m * I) = n} =
  {z : ℂ | abs (z + n * I) - abs (z - m * I) = -m} :=
sorry

end

end same_figure_l89_8995


namespace no_rain_probability_l89_8928

theorem no_rain_probability (p_rain_5th p_rain_6th : ℝ) 
  (h1 : p_rain_5th = 0.2) 
  (h2 : p_rain_6th = 0.4) 
  (h3 : 0 ≤ p_rain_5th ∧ p_rain_5th ≤ 1) 
  (h4 : 0 ≤ p_rain_6th ∧ p_rain_6th ≤ 1) :
  (1 - p_rain_5th) * (1 - p_rain_6th) = 0.48 := by
  sorry

end no_rain_probability_l89_8928


namespace chinese_remainder_theorem_application_l89_8933

theorem chinese_remainder_theorem_application (n : ℤ) : 
  n % 158 = 50 → n % 176 = 66 → n % 16 = 2 := by
  sorry

end chinese_remainder_theorem_application_l89_8933


namespace no_linear_term_implies_a_value_l89_8952

/-- 
Given two polynomials (y + 2a) and (5 - y), if their product does not contain 
a linear term of y, then a = 5/2.
-/
theorem no_linear_term_implies_a_value (a : ℚ) : 
  (∀ y : ℚ, ∃ k m : ℚ, (y + 2*a) * (5 - y) = k*y^2 + m) → a = 5/2 := by
  sorry

end no_linear_term_implies_a_value_l89_8952


namespace sum_of_max_min_is_4032_l89_8994

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ - 2016) ∧
  (∀ x : ℝ, x > 0 → f x > 2016)

/-- The theorem to be proved -/
theorem sum_of_max_min_is_4032 (f : ℝ → ℝ) (h : special_function f) :
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  let N := ⨅ (x : ℝ) (hx : x ∈ Set.Icc (-2016) 2016), f x
  M + N = 4032 := by
  sorry

end sum_of_max_min_is_4032_l89_8994


namespace intersection_M_N_l89_8924

open Set

-- Define set M
def M : Set ℝ := {x : ℝ | (x + 3) * (x - 2) < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem intersection_M_N :
  M ∩ N = Icc 1 2 ∩ Iio 2 :=
sorry

end intersection_M_N_l89_8924


namespace crayons_given_to_friends_l89_8997

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (left : ℕ) 
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332) :
  initial - left - lost = 563 := by
  sorry

end crayons_given_to_friends_l89_8997


namespace max_value_a_l89_8966

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 80) : 
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    b' = 1575 ∧ 
    c' = 394 ∧ 
    d' = 79 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end max_value_a_l89_8966


namespace problem_solution_l89_8900

theorem problem_solution (x y : ℝ) 
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x * y + x + y = 5) :
  x^2 * y + x * y^2 = 4 := by
sorry

end problem_solution_l89_8900


namespace triangle_property_l89_8986

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : 3 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.a = 4 * Real.sqrt 2) : 
  Real.tan t.A = 2 * Real.sqrt 2 ∧ 
  (∃ (S : ℝ), S ≤ 8 * Real.sqrt 2 ∧ 
    ∀ (S' : ℝ), S' = t.a * t.b * Real.sin t.C / 2 → S' ≤ S) := by
  sorry

end triangle_property_l89_8986


namespace tangent_line_sum_l89_8904

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, x = 1 → f x = (1/2) * x + 2) :
  f 1 + (deriv f) 1 = 3 := by
  sorry

end tangent_line_sum_l89_8904


namespace binary_multiplication_division_equality_l89_8931

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem binary_multiplication_division_equality :
  let a := [false, true, true, true, false, true] -- 101110₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, false, true] -- 100₂
  let result_binary := [true, true, false, false, true, true, false, true, true, false, true, true] -- 101110110011₂
  let result_decimal : ℕ := 2995
  (binary_to_decimal a * binary_to_decimal b) / binary_to_decimal c = binary_to_decimal result_binary ∧
  binary_to_decimal result_binary = result_decimal ∧
  decimal_to_binary result_decimal = result_binary :=
by sorry

end binary_multiplication_division_equality_l89_8931


namespace ellipse_equation_for_given_properties_l89_8996

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its properties -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  fun x y => x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating the equation of the ellipse with given properties -/
theorem ellipse_equation_for_given_properties (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.minor_axis_length = 8 * Real.sqrt 2)
  (h4 : e.eccentricity = 1/3) :
  ellipse_equation e = fun x y => x^2 / 36 + y^2 / 32 = 1 := by
  sorry

end ellipse_equation_for_given_properties_l89_8996


namespace triangular_grid_properties_l89_8950

/-- Represents a labeled vertex in the triangular grid -/
structure LabeledVertex where
  x : ℕ
  y : ℕ
  label : ℝ

/-- Represents the triangular grid -/
structure TriangularGrid where
  n : ℕ
  vertices : List LabeledVertex
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for adjacent triangles -/
def adjacent_condition (grid : TriangularGrid) : Prop :=
  ∀ A B C D : LabeledVertex,
    A ∈ grid.vertices → B ∈ grid.vertices → C ∈ grid.vertices → D ∈ grid.vertices →
    (A.x + 1 = B.x ∧ A.y = B.y) →
    (B.x = C.x ∧ B.y + 1 = C.y) →
    (C.x - 1 = D.x ∧ C.y = D.y) →
    A.label + D.label = B.label + C.label

/-- The main theorem -/
theorem triangular_grid_properties (grid : TriangularGrid)
    (h1 : grid.vertices.length = grid.n * (grid.n + 1) / 2)
    (h2 : adjacent_condition grid) :
    (∃ v1 v2 : LabeledVertex,
      v1 ∈ grid.vertices ∧ v2 ∈ grid.vertices ∧
      (∀ v : LabeledVertex, v ∈ grid.vertices → v1.label ≤ v.label ∧ v.label ≤ v2.label) ∧
      ((v1.x - v2.x)^2 + (v1.y - v2.y)^2 : ℝ) = grid.n^2) ∧
    (grid.vertices.map (λ v : LabeledVertex => v.label)).sum =
      (grid.n + 1) * (grid.n + 2) * (grid.a + grid.b + grid.c) / 6 :=
sorry

end triangular_grid_properties_l89_8950


namespace carrie_bought_four_shirts_l89_8944

/-- The number of shirts Carrie bought -/
def num_shirts : ℕ := sorry

/-- The cost of each shirt -/
def shirt_cost : ℕ := 8

/-- The number of pairs of pants Carrie bought -/
def num_pants : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 18

/-- The number of jackets Carrie bought -/
def num_jackets : ℕ := 2

/-- The cost of each jacket -/
def jacket_cost : ℕ := 60

/-- The amount Carrie paid for her half of the clothes -/
def carrie_payment : ℕ := 94

/-- Theorem stating that Carrie bought 4 shirts -/
theorem carrie_bought_four_shirts : num_shirts = 4 := by
  sorry

end carrie_bought_four_shirts_l89_8944


namespace systematic_sampling_l89_8992

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_students : Nat) 
  (num_parts : Nat) 
  (first_part_end : Nat) 
  (first_drawn : Nat) 
  (nth_draw : Nat) :
  total_students = 1000 →
  num_parts = 50 →
  first_part_end = 20 →
  first_drawn = 15 →
  nth_draw = 40 →
  (nth_draw - 1) * (total_students / num_parts) + first_drawn = 795 :=
by
  sorry

end systematic_sampling_l89_8992


namespace nail_decoration_theorem_l89_8920

/-- The time it takes to decorate nails with three coats -/
def nail_decoration_time (application_time dry_time number_of_coats : ℕ) : ℕ :=
  (application_time + dry_time) * number_of_coats

/-- Theorem: The total time to apply and dry three coats on nails is 120 minutes -/
theorem nail_decoration_theorem :
  nail_decoration_time 20 20 3 = 120 :=
by sorry

end nail_decoration_theorem_l89_8920


namespace juice_bottles_count_l89_8953

theorem juice_bottles_count : ∃ x : ℕ, 
  let day0_remaining := x / 2 + 1
  let day1_remaining := day0_remaining / 2
  let day2_remaining := day1_remaining / 2 - 1
  x > 0 ∧ day2_remaining = 2 → x = 22 := by
  sorry

end juice_bottles_count_l89_8953
