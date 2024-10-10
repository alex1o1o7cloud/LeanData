import Mathlib

namespace probability_of_mathematics_letter_l3828_382822

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in "MATHEMATICS" -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from "MATHEMATICS" -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_mathematics_letter :
  probability = 4 / 13 := by sorry

end probability_of_mathematics_letter_l3828_382822


namespace eight_thousand_eight_place_values_l3828_382800

/-- Represents the place value of a digit in a number -/
inductive PlaceValue
  | Ones
  | Tens
  | Hundreds
  | Thousands

/-- Returns the place value of a digit based on its position from the right -/
def getPlaceValue (position : Nat) : PlaceValue :=
  match position with
  | 1 => PlaceValue.Ones
  | 2 => PlaceValue.Tens
  | 3 => PlaceValue.Hundreds
  | 4 => PlaceValue.Thousands
  | _ => PlaceValue.Ones  -- Default to Ones for other positions

/-- Represents a digit in a specific position of a number -/
structure Digit where
  value : Nat
  position : Nat

/-- Theorem: In the number 8008, the 8 in the first position from the right represents 8 units of ones,
    and the 8 in the fourth position from the right represents 8 units of thousands -/
theorem eight_thousand_eight_place_values :
  let num := 8008
  let rightmost_eight : Digit := { value := 8, position := 1 }
  let leftmost_eight : Digit := { value := 8, position := 4 }
  (getPlaceValue rightmost_eight.position = PlaceValue.Ones) ∧
  (getPlaceValue leftmost_eight.position = PlaceValue.Thousands) :=
by sorry

end eight_thousand_eight_place_values_l3828_382800


namespace remainder_of_5n_mod_11_l3828_382884

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end remainder_of_5n_mod_11_l3828_382884


namespace geometric_to_arithmetic_sequence_l3828_382861

theorem geometric_to_arithmetic_sequence (a₁ a₂ a₃ a₄ q : ℝ) : 
  q > 0 ∧ q ≠ 1 ∧
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧
  ((a₁ + a₃ = 2 * a₂) ∨ (a₁ + a₄ = 2 * a₃)) →
  q = ((-1 + Real.sqrt 5) / 2) ∨ q = ((1 + Real.sqrt 5) / 2) := by
sorry

end geometric_to_arithmetic_sequence_l3828_382861


namespace pythagorean_triple_properties_l3828_382806

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def isPrimitivePythagoreanTriple (a b c : ℕ) : Prop :=
  isPythagoreanTriple a b c ∧ Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1

theorem pythagorean_triple_properties
  (a b c : ℕ) (h : isPrimitivePythagoreanTriple a b c) :
  (Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) ∧
  (a % 4 = 0 ∨ b % 4 = 0) ∧
  (a % 3 = 0 ∨ b % 3 = 0) ∧
  (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) ∧
  (∃ k : ℕ, c = 4*k + 1 ∧ c % 3 ≠ 0 ∧ c % 7 ≠ 0 ∧ c % 11 ≠ 0) :=
by
  sorry

end pythagorean_triple_properties_l3828_382806


namespace negation_of_universal_proposition_l3828_382831

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end negation_of_universal_proposition_l3828_382831


namespace exterior_angle_measure_l3828_382836

theorem exterior_angle_measure (a b : ℝ) (ha : a = 40) (hb : b = 30) : 
  180 - (180 - a - b) = 70 := by sorry

end exterior_angle_measure_l3828_382836


namespace unique_digit_sum_l3828_382895

theorem unique_digit_sum (A B C D X Y Z : ℕ) : 
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ X < 10 ∧ Y < 10 ∧ Z < 10) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ X ∧ A ≠ Y ∧ A ≠ Z ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ X ∧ B ≠ Y ∧ B ≠ Z ∧
   C ≠ D ∧ C ≠ X ∧ C ≠ Y ∧ C ≠ Z ∧
   D ≠ X ∧ D ≠ Y ∧ D ≠ Z ∧
   X ≠ Y ∧ X ≠ Z ∧
   Y ≠ Z) →
  (10 * A + B) + (10 * C + D) = 100 * X + 10 * Y + Z →
  Y = X + 1 →
  Z = X + 2 →
  A + B + C + D + X + Y + Z = 24 :=
by sorry

end unique_digit_sum_l3828_382895


namespace equation_solution_l3828_382855

theorem equation_solution : 
  ∃ x₁ x₂ : ℚ, (x₁ = 5/2 ∧ x₂ = -1/2) ∧ 
  (∀ x : ℚ, 4 * (x - 1)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l3828_382855


namespace books_sold_l3828_382848

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) (h1 : initial_books = 242) (h2 : remaining_books = 105) :
  initial_books - remaining_books = 137 := by
sorry

end books_sold_l3828_382848


namespace function_property_l3828_382823

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q : ℝ, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1 * f 1 + f 2) / f 1 + (f 2 * f 2 + f 4) / f 3 + 
  (f 3 * f 3 + f 6) / f 5 + (f 4 * f 4 + f 8) / f 7 = 24 := by
  sorry

end function_property_l3828_382823


namespace pizza_piece_volume_l3828_382886

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/4 →
  diameter = 16 →
  num_pieces = 16 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

end pizza_piece_volume_l3828_382886


namespace shopkeeper_cloth_cost_price_l3828_382814

/-- Given a shopkeeper sells cloth at a loss, calculate the cost price per meter. -/
theorem shopkeeper_cloth_cost_price
  (total_meters : ℕ)
  (total_selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  total_selling_price / total_meters + loss_per_meter = 50 := by
  sorry

#check shopkeeper_cloth_cost_price

end shopkeeper_cloth_cost_price_l3828_382814


namespace parabola_axis_of_symmetry_l3828_382868

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -2 * x^2

/-- The equation of the axis of symmetry -/
def axis_of_symmetry (y : ℝ) : Prop := y = 1/8

/-- Theorem: The axis of symmetry for the parabola y = -2x^2 is y = 1/8 -/
theorem parabola_axis_of_symmetry :
  ∀ x y : ℝ, parabola_equation x y → axis_of_symmetry y := by
  sorry

end parabola_axis_of_symmetry_l3828_382868


namespace eighth_odd_multiple_of_five_l3828_382842

def is_odd_multiple_of_five (n : ℕ) : Prop := n % 2 = 1 ∧ n % 5 = 0

def nth_odd_multiple_of_five (n : ℕ) : ℕ :=
  (2 * n - 1) * 5

theorem eighth_odd_multiple_of_five :
  nth_odd_multiple_of_five 8 = 75 ∧ is_odd_multiple_of_five (nth_odd_multiple_of_five 8) :=
sorry

end eighth_odd_multiple_of_five_l3828_382842


namespace half_power_inequality_l3828_382845

theorem half_power_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by sorry

end half_power_inequality_l3828_382845


namespace lcm_18_35_l3828_382897

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end lcm_18_35_l3828_382897


namespace emily_remaining_toys_l3828_382803

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

theorem emily_remaining_toys : remaining_toys = 4 := by
  sorry

end emily_remaining_toys_l3828_382803


namespace parallel_vectors_sum_magnitude_l3828_382857

/-- Given two parallel vectors p and q, prove that their sum has magnitude √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (h_parallel : p.1 * q.2 = p.2 * q.1) :
  p = (2, -3) → q.2 = 6 → ‖p + q‖ = Real.sqrt 13 := by
  sorry

end parallel_vectors_sum_magnitude_l3828_382857


namespace point_on_line_coordinates_l3828_382841

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line passing through two points in 3D space -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Function to get a point on a line given an x-coordinate -/
def pointOnLine (l : Line3D) (x : ℝ) : Point3D :=
  sorry

theorem point_on_line_coordinates (l : Line3D) :
  l.p1 = ⟨1, 3, 4⟩ →
  l.p2 = ⟨4, 2, 1⟩ →
  let p := pointOnLine l 7
  p.y = 1 ∧ p.z = -2 := by
  sorry

end point_on_line_coordinates_l3828_382841


namespace distinct_domino_arrangements_l3828_382877

/-- Represents a grid with width and height -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Calculates the number of distinct paths on a grid using a given number of dominoes -/
def countDistinctPaths (g : Grid) (d : Domino) (numDominoes : Nat) : Nat :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem: The number of distinct domino arrangements on a 6x5 grid with 5 dominoes is 126 -/
theorem distinct_domino_arrangements :
  let g : Grid := ⟨6, 5⟩
  let d : Domino := ⟨2, 1⟩
  countDistinctPaths g d 5 = 126 := by
  sorry

#eval countDistinctPaths ⟨6, 5⟩ ⟨2, 1⟩ 5

end distinct_domino_arrangements_l3828_382877


namespace tea_canister_production_balance_l3828_382882

/-- Represents the production balance in a factory producing cylindrical tea canisters -/
theorem tea_canister_production_balance 
  (total_workers : ℕ) 
  (bodies_per_hour : ℕ) 
  (bottoms_per_hour : ℕ) 
  (bottoms_per_body : ℕ) 
  (body_workers : ℕ) :
  total_workers = 44 →
  bodies_per_hour = 50 →
  bottoms_per_hour = 120 →
  bottoms_per_body = 2 →
  body_workers ≤ total_workers →
  (2 * bottoms_per_hour * (total_workers - body_workers) = bodies_per_hour * body_workers) ↔
  (bottoms_per_body * bottoms_per_hour * (total_workers - body_workers) = bodies_per_hour * body_workers) :=
by sorry

end tea_canister_production_balance_l3828_382882


namespace divisor_difference_two_l3828_382885

theorem divisor_difference_two (k : ℕ+) :
  ∃ (m : ℕ) (d : Fin (m + 1) → ℕ),
    (∀ i, d i ∣ (4 * k)) ∧
    (d 0 = 1) ∧
    (d (Fin.last m) = 4 * k) ∧
    (∀ i j, i < j → d i < d j) ∧
    (∃ i : Fin m, d i.succ - d i = 2) :=
by sorry

end divisor_difference_two_l3828_382885


namespace max_area_height_l3828_382829

/-- A right trapezoid with an acute angle of 30° and perimeter 6 -/
structure RightTrapezoid where
  height : ℝ
  sumOfBases : ℝ
  acuteAngle : ℝ
  perimeter : ℝ
  area : ℝ
  acuteAngle_eq : acuteAngle = π / 6
  perimeter_eq : perimeter = 6
  area_eq : area = (3 * sumOfBases * height) / 2
  perimeter_constraint : sumOfBases + 3 * height = 6

/-- The height that maximizes the area of the right trapezoid is 1 -/
theorem max_area_height (t : RightTrapezoid) : 
  t.area ≤ (3 : ℝ) / 2 ∧ (t.area = (3 : ℝ) / 2 ↔ t.height = 1) :=
sorry

end max_area_height_l3828_382829


namespace solution_comparison_l3828_382866

theorem solution_comparison (c d p q : ℝ) (hc : c ≠ 0) (hp : p ≠ 0) :
  -d / c < -q / p ↔ q / p < d / c := by sorry

end solution_comparison_l3828_382866


namespace gift_card_value_l3828_382818

theorem gift_card_value (coffee_price : ℝ) (pounds_bought : ℝ) (remaining_balance : ℝ) :
  coffee_price = 8.58 →
  pounds_bought = 4 →
  remaining_balance = 35.68 →
  coffee_price * pounds_bought + remaining_balance = 70 :=
by
  sorry

end gift_card_value_l3828_382818


namespace total_profit_calculation_l3828_382810

/-- Prove that the total profit is 60000 given the investment ratios and C's profit share -/
theorem total_profit_calculation (a b c : ℕ) (total_profit : ℕ) : 
  a * 2 = c * 3 →  -- A and C invested in ratio 3:2
  a = b * 3 →      -- A and B invested in ratio 3:1
  c * total_profit = 20000 * (a + b + c) →  -- C's profit share
  total_profit = 60000 := by
  sorry

#check total_profit_calculation

end total_profit_calculation_l3828_382810


namespace express_y_in_terms_of_x_l3828_382811

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 3 * x = 5) : y = 3 * x + 5 := by
  sorry

end express_y_in_terms_of_x_l3828_382811


namespace legos_lost_l3828_382817

theorem legos_lost (initial_legos current_legos : ℕ) 
  (h1 : initial_legos = 380) 
  (h2 : current_legos = 323) : 
  initial_legos - current_legos = 57 := by
  sorry

end legos_lost_l3828_382817


namespace amanda_notebooks_l3828_382850

/-- Calculates the final number of notebooks Amanda has -/
def final_notebooks (initial : ℕ) (ordered : ℕ) (lost : ℕ) : ℕ :=
  initial + ordered - lost

theorem amanda_notebooks :
  final_notebooks 10 6 2 = 14 :=
by sorry

end amanda_notebooks_l3828_382850


namespace inverse_variation_problem_l3828_382860

/-- Given that a² and √b vary inversely, a = 3 when b = 36, and ab = 108, prove that b = 36 -/
theorem inverse_variation_problem (a b : ℝ) (h1 : ∃ k : ℝ, a^2 * Real.sqrt b = k)
  (h2 : a = 3 ∧ b = 36) (h3 : a * b = 108) : b = 36 := by
  sorry

end inverse_variation_problem_l3828_382860


namespace quadratic_equation_roots_l3828_382871

theorem quadratic_equation_roots : 
  let equation := fun (x : ℂ) => x^2 + x + 2
  ∃ (r₁ r₂ : ℂ), r₁ = (-1 + Complex.I * Real.sqrt 7) / 2 ∧ 
                  r₂ = (-1 - Complex.I * Real.sqrt 7) / 2 ∧ 
                  equation r₁ = 0 ∧ 
                  equation r₂ = 0 ∧
                  ∀ (x : ℂ), equation x = 0 → x = r₁ ∨ x = r₂ :=
by sorry

end quadratic_equation_roots_l3828_382871


namespace line_points_l3828_382825

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

/-- The main theorem -/
theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, 4⟩
  let points_on_line : List Point := [⟨5, 10⟩, ⟨7, 14⟩, ⟨10, 20⟩, ⟨3, 6⟩]
  let point_not_on_line : Point := ⟨4, 7⟩
  (∀ p ∈ points_on_line, lies_on_line p p1 p2) ∧
  ¬(lies_on_line point_not_on_line p1 p2) := by
  sorry

end line_points_l3828_382825


namespace friend_reading_time_l3828_382898

theorem friend_reading_time (my_time : ℝ) (friend_speed_multiplier : ℝ) (distraction_time : ℝ) :
  my_time = 1.5 →
  friend_speed_multiplier = 5 →
  distraction_time = 0.25 →
  (my_time * 60) / friend_speed_multiplier + distraction_time = 33 :=
by sorry

end friend_reading_time_l3828_382898


namespace quadrilateral_ratio_theorem_l3828_382881

-- Define the quadrilateral and points
variable (A B C D K L M N P : ℝ × ℝ)
variable (α β : ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def point_on_side (X Y Z : ℝ × ℝ) : Prop := sorry

def ratio_equals (A B X Y : ℝ × ℝ) (r : ℝ) : Prop := sorry

def intersection_point (K M L N P : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_ratio_theorem 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_K : point_on_side K A B)
  (h_L : point_on_side L B C)
  (h_M : point_on_side M C D)
  (h_N : point_on_side N D A)
  (h_AK_KB : ratio_equals A K K B α)
  (h_DM_MC : ratio_equals D M M C α)
  (h_BL_LC : ratio_equals B L L C β)
  (h_AN_ND : ratio_equals A N N D β)
  (h_P : intersection_point K M L N P) :
  ratio_equals N P P L α ∧ ratio_equals K P P M β := by sorry

end quadrilateral_ratio_theorem_l3828_382881


namespace track_circumference_l3828_382851

/-- Represents the circular track and the runners' positions --/
structure TrackSystem where
  circumference : ℝ
  first_meeting_distance : ℝ
  second_meeting_distance : ℝ

/-- The conditions of the problem --/
def problem_conditions (t : TrackSystem) : Prop :=
  t.first_meeting_distance = 150 ∧
  t.second_meeting_distance = t.circumference - 90 ∧
  2 * t.circumference = t.first_meeting_distance * 2 + t.second_meeting_distance

/-- The theorem stating that the circumference is 300 yards --/
theorem track_circumference (t : TrackSystem) :
  problem_conditions t → t.circumference = 300 :=
by
  sorry


end track_circumference_l3828_382851


namespace triangle_area_equivalence_l3828_382834

/-- Given a triangle with angles α, β, γ, side length a opposite to angle α,
    and circumradius R, prove that the two expressions for the area S are equivalent. -/
theorem triangle_area_equivalence (α β γ a R : ℝ) (h_angles : α + β + γ = π)
    (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < a ∧ 0 < R) :
  (a^2 * Real.sin β * Real.sin γ) / (2 * Real.sin α) =
  2 * R^2 * Real.sin α * Real.sin β * Real.sin γ := by
sorry

end triangle_area_equivalence_l3828_382834


namespace cubic_expression_equality_l3828_382827

theorem cubic_expression_equality : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end cubic_expression_equality_l3828_382827


namespace rationalize_denominator_l3828_382833

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 6 + 5 * Real.sqrt 7) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 6 ∧ C = 15 ∧ D = 7 ∧ E = 79 ∧
    Int.gcd A E = 1 ∧ Int.gcd C E = 1 ∧
    Int.gcd B D = 1 :=
by sorry

end rationalize_denominator_l3828_382833


namespace gcd_of_B_is_two_l3828_382843

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l3828_382843


namespace parallel_vectors_m_value_l3828_382807

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (1, m - 3/2)
  are_parallel a b → m = 0 := by
  sorry

end parallel_vectors_m_value_l3828_382807


namespace tangent_line_equality_l3828_382876

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^3 + 2*a*x^2 + b*x + a
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Define the derivatives of f and g
def f' (a b x : ℝ) : ℝ := 3*x^2 + 4*a*x + b
def g' (x : ℝ) : ℝ := 2*x - 3

-- State the theorem
theorem tangent_line_equality (a b : ℝ) :
  f a b 2 = g 2 ∧ f' a b 2 = g' 2 →
  a = -2 ∧ b = 5 ∧ ∀ x y, y = x - 2 ↔ x - y - 2 = 0 :=
by sorry

end tangent_line_equality_l3828_382876


namespace translation_theorem_l3828_382863

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C.x = 4 ∧ C.y = 7 ∧
  C = applyTranslation t A →
  applyTranslation t B = { x := 1, y := 2 } := by
  sorry


end translation_theorem_l3828_382863


namespace at_least_one_subgraph_not_planar_l3828_382865

/-- A complete graph with 11 vertices where each edge is colored either red or blue. -/
def CompleteGraph11 : Type := Unit

/-- The red subgraph of the complete graph. -/
def RedSubgraph (G : CompleteGraph11) : Type := Unit

/-- The blue subgraph of the complete graph. -/
def BlueSubgraph (G : CompleteGraph11) : Type := Unit

/-- Predicate to check if a graph is planar. -/
def IsPlanar (G : Type) : Prop := sorry

/-- Theorem stating that at least one of the monochromatic subgraphs is not planar. -/
theorem at_least_one_subgraph_not_planar (G : CompleteGraph11) : 
  ¬(IsPlanar (RedSubgraph G) ∧ IsPlanar (BlueSubgraph G)) := by
  sorry

end at_least_one_subgraph_not_planar_l3828_382865


namespace base_8_digit_count_l3828_382815

/-- The count of numbers among the first 512 positive integers in base 8 
    that contain either 5 or 6 -/
def count_with_5_or_6 : ℕ := 296

/-- The count of numbers among the first 512 positive integers in base 8 
    that don't contain 5 or 6 -/
def count_without_5_or_6 : ℕ := 6^3

/-- The total count of numbers considered -/
def total_count : ℕ := 512

theorem base_8_digit_count : 
  count_with_5_or_6 = total_count - count_without_5_or_6 := by sorry

end base_8_digit_count_l3828_382815


namespace museum_visit_permutations_l3828_382832

theorem museum_visit_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end museum_visit_permutations_l3828_382832


namespace passing_percentage_is_33_percent_l3828_382830

def total_marks : ℕ := 400
def obtained_marks : ℕ := 92
def failing_margin : ℕ := 40

theorem passing_percentage_is_33_percent :
  (obtained_marks + failing_margin) / total_marks * 100 = 33 := by sorry

end passing_percentage_is_33_percent_l3828_382830


namespace inequality_proof_l3828_382872

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d)
  (hsum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end inequality_proof_l3828_382872


namespace ellipse_k_range_l3828_382824

def is_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    x^2 / (k - 4) + y^2 / (9 - k) = 1 ↔ (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_k_range (k : ℝ) :
  is_ellipse k ↔ (k ∈ Set.Ioo 4 (13/2) ∪ Set.Ioo (13/2) 9) :=
sorry

end ellipse_k_range_l3828_382824


namespace shirt_pricing_solution_l3828_382891

/-- Represents the shirt pricing problem with given conditions --/
structure ShirtPricingProblem where
  cost_price : ℝ
  initial_sales : ℝ
  initial_profit_per_shirt : ℝ
  price_reduction_effect : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily sales based on the price reduction --/
def daily_sales (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  p.initial_sales + p.price_reduction_effect * (p.cost_price + p.initial_profit_per_shirt - selling_price)

/-- Calculates the daily profit based on the selling price --/
def daily_profit (p : ShirtPricingProblem) (selling_price : ℝ) : ℝ :=
  (selling_price - p.cost_price) * (daily_sales p selling_price)

/-- Theorem stating that the selling price should be either $105 or $120 --/
theorem shirt_pricing_solution (p : ShirtPricingProblem)
  (h1 : p.cost_price = 80)
  (h2 : p.initial_sales = 30)
  (h3 : p.initial_profit_per_shirt = 50)
  (h4 : p.price_reduction_effect = 2)
  (h5 : p.target_daily_profit = 2000) :
  ∃ (x : ℝ), (x = 105 ∨ x = 120) ∧ daily_profit p x = p.target_daily_profit :=
sorry

end shirt_pricing_solution_l3828_382891


namespace remaining_work_time_is_three_l3828_382899

/-- The time taken by A to finish the remaining work after B has worked for 10 days -/
def remaining_work_time (a_time b_time b_worked_days : ℚ) : ℚ :=
  let b_work_rate := 1 / b_time
  let b_work_done := b_work_rate * b_worked_days
  let remaining_work := 1 - b_work_done
  let a_work_rate := 1 / a_time
  remaining_work / a_work_rate

/-- Theorem stating that A will take 3 days to finish the remaining work -/
theorem remaining_work_time_is_three :
  remaining_work_time 9 15 10 = 3 := by
  sorry

#eval remaining_work_time 9 15 10

end remaining_work_time_is_three_l3828_382899


namespace circle_center_correct_l3828_382828

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The Cartesian equation of a circle -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 = 4 * y

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (0, 2)

theorem circle_center_correct :
  (∀ ρ θ : ℝ, polar_equation ρ θ ↔ ∃ x y : ℝ, cartesian_equation x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ x y : ℝ, cartesian_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 4) :=
sorry

end circle_center_correct_l3828_382828


namespace password_count_l3828_382859

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- 10 digits minus 1 (7 is excluded)

def total_passwords : ℕ := available_digits ^ password_length

def all_different_passwords : ℕ := Nat.choose available_digits password_length * Nat.factorial password_length

theorem password_count : 
  total_passwords - all_different_passwords = 3537 :=
by sorry

end password_count_l3828_382859


namespace rhombus_perimeter_l3828_382846

/-- The perimeter of a rhombus with diagonals 18 and 12 is 12√13 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 12) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 12 * Real.sqrt 13 := by
  sorry

end rhombus_perimeter_l3828_382846


namespace fraction_equality_l3828_382839

theorem fraction_equality (a b : ℝ) (h : a ≠ b) : (-a + b) / (a - b) = -1 := by
  sorry

end fraction_equality_l3828_382839


namespace monotone_decreasing_range_l3828_382808

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2*a - 1)*x + a
  else Real.log x / Real.log a

-- State the theorem
theorem monotone_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ↔ 0 < a ∧ a ≤ 1/3 :=
sorry

end monotone_decreasing_range_l3828_382808


namespace parallel_transitive_perpendicular_from_line_l3828_382887

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem for proposition ①
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel α γ → parallel γ β := by sorry

-- Theorem for proposition ③
theorem perpendicular_from_line (m : Line) (α β : Plane) :
  line_perpendicular m α → line_parallel m β → perpendicular α β := by sorry

end parallel_transitive_perpendicular_from_line_l3828_382887


namespace prob_sum_14_four_dice_l3828_382847

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 14

/-- The total number of possible outcomes when rolling four dice -/
def total_outcomes : ℕ := faces ^ num_dice

/-- The number of favorable outcomes (sum of 14) -/
def favorable_outcomes : ℕ := 54

/-- The probability of rolling a sum of 14 with four standard six-faced dice -/
theorem prob_sum_14_four_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 54 / 1296 := by sorry

end prob_sum_14_four_dice_l3828_382847


namespace chess_tournament_theorem_l3828_382849

/-- Represents a single-elimination chess tournament -/
structure ChessTournament where
  participants : ℕ
  winner_games : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n
  winner_played_six : winner_games = 6

/-- Number of participants who won at least 2 more games than they lost -/
def participants_with_two_more_wins (t : ChessTournament) : ℕ :=
  8

theorem chess_tournament_theorem (t : ChessTournament) :
  participants_with_two_more_wins t = 8 := by
  sorry

end chess_tournament_theorem_l3828_382849


namespace binomial_expansion_sum_l3828_382893

/-- Given that (1 - 2/x)³ = a₀ + a₁·(1/x) + a₂·(1/x)² + a₃·(1/x)³, prove that a₁ + a₂ = 6 -/
theorem binomial_expansion_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁ * (1/x) + a₂ * (1/x)^2 + a₃ * (1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end binomial_expansion_sum_l3828_382893


namespace julie_school_work_hours_l3828_382826

-- Define the given parameters
def summer_weeks : ℕ := 10
def summer_hours_per_week : ℕ := 36
def summer_earnings : ℕ := 4500
def school_weeks : ℕ := 45
def school_earnings : ℕ := 4500

-- Define the function to calculate required hours per week
def required_hours_per_week (weeks : ℕ) (total_earnings : ℕ) (hourly_rate : ℚ) : ℚ :=
  (total_earnings : ℚ) / (weeks : ℚ) / hourly_rate

-- Theorem statement
theorem julie_school_work_hours :
  let hourly_rate : ℚ := (summer_earnings : ℚ) / ((summer_weeks * summer_hours_per_week) : ℚ)
  required_hours_per_week school_weeks school_earnings hourly_rate = 8 := by
  sorry

end julie_school_work_hours_l3828_382826


namespace smallest_solution_of_equation_l3828_382856

theorem smallest_solution_of_equation (x : ℝ) :
  (((3 * x) / (x - 3)) + ((3 * x^2 - 27) / x) = 15) →
  (x ≥ -1 ∧ (∀ y : ℝ, y < -1 → ((3 * y) / (y - 3)) + ((3 * y^2 - 27) / y) ≠ 15)) :=
by sorry

end smallest_solution_of_equation_l3828_382856


namespace correct_systematic_sampling_l3828_382838

/-- Represents the systematic sampling of students -/
def systematic_sampling (total_students : ℕ) (students_to_select : ℕ) : List ℕ :=
  sorry

/-- The theorem stating the correct systematic sampling for the given problem -/
theorem correct_systematic_sampling :
  systematic_sampling 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end correct_systematic_sampling_l3828_382838


namespace negation_of_proposition_l3828_382864

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) := by
  sorry

end negation_of_proposition_l3828_382864


namespace x_equals_negative_x_and_abs_x_equals_two_l3828_382878

theorem x_equals_negative_x_and_abs_x_equals_two (x : ℝ) :
  (x = -x → x = 0) ∧ (|x| = 2 → x = 2 ∨ x = -2) := by
  sorry

end x_equals_negative_x_and_abs_x_equals_two_l3828_382878


namespace fraction_equality_unique_solution_l3828_382853

theorem fraction_equality_unique_solution :
  ∃! (C D : ℝ), ∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (D * x - 23) / (x^2 - 9*x + 20) = C / (x - 4) + 7 / (x - 5) :=
by
  -- The proof goes here
  sorry

end fraction_equality_unique_solution_l3828_382853


namespace average_age_of_class_l3828_382862

theorem average_age_of_class (total_students : ℕ) 
  (group1_count group2_count : ℕ) 
  (group1_avg group2_avg last_student_age : ℝ) : 
  total_students = group1_count + group2_count + 1 →
  group1_count = 8 →
  group2_count = 6 →
  group1_avg = 14 →
  group2_avg = 16 →
  last_student_age = 17 →
  (group1_count * group1_avg + group2_count * group2_avg + last_student_age) / total_students = 15 :=
by sorry

end average_age_of_class_l3828_382862


namespace sculpture_cost_brl_l3828_382883

/-- Exchange rate from USD to AUD -/
def usd_to_aud : ℝ := 5

/-- Exchange rate from USD to BRL -/
def usd_to_brl : ℝ := 10

/-- Cost of the sculpture in AUD -/
def sculpture_cost_aud : ℝ := 200

/-- Theorem stating the equivalent cost of the sculpture in BRL -/
theorem sculpture_cost_brl : 
  (sculpture_cost_aud / usd_to_aud) * usd_to_brl = 400 := by
  sorry

end sculpture_cost_brl_l3828_382883


namespace steve_pie_difference_l3828_382809

/-- The number of days Steve bakes apple pies in a week -/
def apple_pie_days : ℕ := 3

/-- The number of days Steve bakes cherry pies in a week -/
def cherry_pie_days : ℕ := 2

/-- The number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The number of apple pies Steve bakes in a week -/
def apple_pies_per_week : ℕ := apple_pie_days * pies_per_day

/-- The number of cherry pies Steve bakes in a week -/
def cherry_pies_per_week : ℕ := cherry_pie_days * pies_per_day

theorem steve_pie_difference : apple_pies_per_week - cherry_pies_per_week = 12 := by
  sorry

end steve_pie_difference_l3828_382809


namespace simplify_fraction_l3828_382840

theorem simplify_fraction (a : ℚ) (h : a = 2) : 24 * a^5 / (72 * a^3) = 4/3 := by
  sorry

end simplify_fraction_l3828_382840


namespace circle_condition_l3828_382875

/-- The equation of a potential circle with a parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 4*y + m = 0

/-- A predicate to check if an equation represents a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) : is_circle m ↔ m < 5 := by
  sorry

end circle_condition_l3828_382875


namespace find_divisor_l3828_382821

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 13787)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 155 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 155 := by
  sorry

end find_divisor_l3828_382821


namespace function_characterization_l3828_382894

-- Define the property that the function f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f a + b) * (a - f b)

-- State the theorem
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end function_characterization_l3828_382894


namespace cylinder_radius_problem_l3828_382858

theorem cylinder_radius_problem (r : ℝ) (y : ℝ) : 
  r > 0 →
  (π * ((r + 4)^2 * 4 - r^2 * 4) = y) →
  (π * (r^2 * 8 - r^2 * 4) = y) →
  r = 2 + 2 * Real.sqrt 2 :=
by sorry

end cylinder_radius_problem_l3828_382858


namespace second_to_last_digit_is_five_l3828_382869

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

theorem second_to_last_digit_is_five (N : ℕ) 
  (h1 : N % 10 = 0) 
  (h2 : ∃ d : ℕ, d < N ∧ d ∣ N ∧ is_power_of_prime d ∧ ∀ m : ℕ, m < N → m ∣ N → m ≤ d)
  (h3 : N > 10) :
  (N / 10) % 10 = 5 :=
sorry

end second_to_last_digit_is_five_l3828_382869


namespace remainder_theorem_polynomial_remainder_l3828_382819

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 3

-- State the theorem
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := by sorry

-- State the problem
theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x + 27 := by sorry

end remainder_theorem_polynomial_remainder_l3828_382819


namespace greatest_of_three_consecutive_integers_l3828_382805

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 21) : 
  max x (max (x + 1) (x + 2)) = 8 := by
sorry

end greatest_of_three_consecutive_integers_l3828_382805


namespace max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l3828_382835

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format digital watch display is 24 -/
theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
sorry

/-- Theorem stating that there exists a time that achieves the maximum sum of digits -/
theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
sorry

end max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l3828_382835


namespace cost_difference_70_copies_l3828_382874

/-- Calculates the cost for color copies at print shop X -/
def costX (copies : ℕ) : ℚ :=
  if copies ≤ 50 then
    1.2 * copies
  else
    1.2 * 50 + 0.9 * (copies - 50)

/-- Calculates the cost for color copies at print shop Y -/
def costY (copies : ℕ) : ℚ :=
  10 + 1.7 * copies

/-- The difference in cost between print shop Y and X for 70 color copies is $51 -/
theorem cost_difference_70_copies : costY 70 - costX 70 = 51 := by
  sorry

end cost_difference_70_copies_l3828_382874


namespace smallest_angle_for_sin_polar_graph_l3828_382820

def completes_intrinsic_pattern (t : Real) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ r, r = Real.sin θ ∧ 
  (∀ ϕ, ϕ > t → ∃ ψ, 0 ≤ ψ ∧ ψ ≤ t ∧ Real.sin ϕ = Real.sin ψ)

theorem smallest_angle_for_sin_polar_graph :
  (∀ t < 2 * Real.pi, ¬ completes_intrinsic_pattern t) ∧
  completes_intrinsic_pattern (2 * Real.pi) := by
sorry

end smallest_angle_for_sin_polar_graph_l3828_382820


namespace problem_solution_l3828_382889

theorem problem_solution (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end problem_solution_l3828_382889


namespace smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l3828_382892

theorem smallest_square_with_40_and_49 :
  ∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000 :=
by sorry

theorem existence_of_2000_square :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b :=
by sorry

theorem smallest_2000_square :
  (∀ n : ℕ, 
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n * n = 40 * 40 * a + 49 * 49 * b) →
    n ≥ 2000) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2000 * 2000 = 40 * 40 * a + 49 * 49 * b) :=
by sorry

end smallest_square_with_40_and_49_existence_of_2000_square_smallest_2000_square_l3828_382892


namespace prove_sales_tax_percentage_l3828_382844

def total_spent : ℝ := 184.80
def tip_percentage : ℝ := 20
def food_price : ℝ := 140

def sales_tax_percentage : ℝ := 10

theorem prove_sales_tax_percentage :
  let price_with_tax := food_price * (1 + sales_tax_percentage / 100)
  let total_with_tip := price_with_tax * (1 + tip_percentage / 100)
  total_with_tip = total_spent :=
by sorry

end prove_sales_tax_percentage_l3828_382844


namespace eight_bead_bracelet_arrangements_l3828_382896

/-- The number of unique ways to place n distinct beads on a rotatable, non-flippable bracelet -/
def braceletArrangements (n : ℕ) : ℕ := Nat.factorial n / n

/-- Theorem: The number of unique ways to place 8 distinct beads on a bracelet
    that can be rotated but not flipped is 5040 -/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 5040 := by
  sorry

end eight_bead_bracelet_arrangements_l3828_382896


namespace intersection_point_P_equation_l3828_382837

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3
def C₂ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

-- Theorem for the intersection point
theorem intersection_point :
  ∃ ρ θ, C₁ ρ θ ∧ C₂ ρ θ ∧ ρ = 2 * Real.sqrt 3 ∧ θ = Real.pi / 6 :=
sorry

-- Define the relationship between Q and P
def Q_P_relation (ρ_Q θ_Q ρ_P θ_P : ℝ) : Prop :=
  C₂ ρ_Q θ_Q ∧ ρ_Q = (2/3) * ρ_P ∧ θ_Q = θ_P

-- Theorem for the polar coordinate equation of P
theorem P_equation :
  ∀ ρ_P θ_P, (∃ ρ_Q θ_Q, Q_P_relation ρ_Q θ_Q ρ_P θ_P) →
  ρ_P = 10 * Real.cos θ_P ∧ 0 ≤ θ_P ∧ θ_P < Real.pi / 2 :=
sorry

end intersection_point_P_equation_l3828_382837


namespace paityn_red_hats_l3828_382812

/-- Proves that Paityn has 20 red hats given the problem conditions -/
theorem paityn_red_hats :
  ∀ (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ),
  paityn_blue = 24 →
  zola_red = (4 * paityn_red) / 5 →
  zola_blue = 2 * paityn_blue →
  paityn_red + paityn_blue + zola_red + zola_blue = 108 →
  paityn_red = 20 := by
sorry


end paityn_red_hats_l3828_382812


namespace sequence_nth_term_l3828_382801

theorem sequence_nth_term (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h_sum : ∀ k : ℕ+, S k = k^2) :
  a n = 2 * n - 1 := by
  sorry

end sequence_nth_term_l3828_382801


namespace base6_addition_theorem_l3828_382880

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 represented as a list of digits -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc else aux (n / 6) ((n % 6) :: acc)
    aux n []

/-- Adds two base 6 numbers represented as lists of digits -/
def addBase6 (a b : List Nat) : List Nat :=
  let sum := base6ToBase10 a + base6ToBase10 b
  base10ToBase6 sum

theorem base6_addition_theorem :
  let a := [2, 4, 5, 3]  -- 2453₆
  let b := [1, 6, 4, 3, 2]  -- 16432₆
  addBase6 a b = [2, 5, 5, 4, 5] ∧  -- 25545₆
  base6ToBase10 (addBase6 a b) = 3881 := by
  sorry

end base6_addition_theorem_l3828_382880


namespace marbles_cost_l3828_382813

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : total_spent - (football_cost + baseball_cost) = 9.05 := by
  sorry

end marbles_cost_l3828_382813


namespace smallest_of_three_consecutive_sum_30_l3828_382804

theorem smallest_of_three_consecutive_sum_30 (x : ℕ) :
  x + (x + 1) + (x + 2) = 30 → x = 9 := by
  sorry

end smallest_of_three_consecutive_sum_30_l3828_382804


namespace N_cannot_be_2_7_l3828_382816

def M : Set ℕ := {1, 4, 7}

theorem N_cannot_be_2_7 (N : Set ℕ) (h : M ∪ N = M) : N ≠ {2, 7} := by
  sorry

end N_cannot_be_2_7_l3828_382816


namespace square_root_squared_l3828_382854

theorem square_root_squared (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by
  sorry

end square_root_squared_l3828_382854


namespace pool_capacity_l3828_382852

theorem pool_capacity (initial_percentage : ℚ) (final_percentage : ℚ) (added_water : ℚ) :
  initial_percentage = 0.4 →
  final_percentage = 0.8 →
  added_water = 300 →
  (∃ (total_capacity : ℚ), 
    total_capacity * final_percentage = total_capacity * initial_percentage + added_water ∧
    total_capacity = 750) :=
by sorry

end pool_capacity_l3828_382852


namespace school_population_l3828_382879

/-- Given a school with boys, girls, and teachers, prove that the total number
    of people is 41b/32 when there are 4 times as many boys as girls and 8 times
    as many girls as teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end school_population_l3828_382879


namespace min_segment_length_in_right_angle_l3828_382802

/-- Given a point inside a 90° angle, located 8 units from one side and 1 unit from the other side,
    the minimum length of a segment passing through this point with ends on the sides of the angle is 10 units. -/
theorem min_segment_length_in_right_angle (P : ℝ × ℝ) 
  (inside_angle : P.1 > 0 ∧ P.2 > 0) 
  (dist_to_sides : P.1 = 1 ∧ P.2 = 8) : 
  Real.sqrt ((P.1 + P.1)^2 + (P.2 + P.2)^2) = 10 := by
  sorry

end min_segment_length_in_right_angle_l3828_382802


namespace equation_solution_l3828_382867

theorem equation_solution : ∃! x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3 ∧ x = -1 := by
  sorry

end equation_solution_l3828_382867


namespace sum_equals_negative_seven_and_half_l3828_382870

/-- Given that p + 2 = q + 3 = r + 4 = s + 5 = t + 6 = p + q + r + s + t + 10,
    prove that p + q + r + s + t = -7.5 -/
theorem sum_equals_negative_seven_and_half
  (p q r s t : ℚ)
  (h : p + 2 = q + 3 ∧ 
       q + 3 = r + 4 ∧ 
       r + 4 = s + 5 ∧ 
       s + 5 = t + 6 ∧ 
       t + 6 = p + q + r + s + t + 10) :
  p + q + r + s + t = -7.5 := by
sorry

end sum_equals_negative_seven_and_half_l3828_382870


namespace chord_length_theorem_l3828_382888

/-- In a right triangle ABC with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of leg AB -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- a and r are positive -/
  a_pos : 0 < a
  r_pos : 0 < r

/-- The chord length theorem -/
theorem chord_length_theorem (t : RightTriangleWithInscribedCircle) :
  ∃ (chord_length : ℝ),
    chord_length = (2 * t.a * t.r) / Real.sqrt (t.a^2 + t.r^2) :=
by sorry

end chord_length_theorem_l3828_382888


namespace polynomial_not_equal_33_l3828_382873

theorem polynomial_not_equal_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end polynomial_not_equal_33_l3828_382873


namespace problem_solution_l3828_382890

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem problem_solution :
  ∀ t : ℝ,
  (∀ x : ℝ, f t x ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) →
  (t = 1 ∧
   ∀ a b c : ℝ,
   a > 0 → b > 0 → c > 0 →
   a + b + c = t →
   a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end problem_solution_l3828_382890
