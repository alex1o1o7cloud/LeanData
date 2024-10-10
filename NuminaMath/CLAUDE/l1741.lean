import Mathlib

namespace sunzi_wood_problem_l1741_174134

theorem sunzi_wood_problem (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ (rope / 2) + 1 = x) → 
  (1/2 * (x + 4.5) = x - 1) := by
sorry

end sunzi_wood_problem_l1741_174134


namespace system_solution_l1741_174142

theorem system_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end system_solution_l1741_174142


namespace hydrogen_oxygen_reaction_certain_l1741_174153

/-- Represents the certainty of an event --/
inductive EventCertainty
  | Possible
  | Impossible
  | Certain

/-- Represents a chemical reaction --/
structure ChemicalReaction where
  reactants : List String
  products : List String

/-- The chemical reaction of hydrogen burning in oxygen to form water --/
def hydrogenOxygenReaction : ChemicalReaction :=
  { reactants := ["Hydrogen", "Oxygen"],
    products := ["Water"] }

/-- Theorem stating that the hydrogen-oxygen reaction is certain --/
theorem hydrogen_oxygen_reaction_certain :
  (hydrogenOxygenReaction.reactants = ["Hydrogen", "Oxygen"] ∧
   hydrogenOxygenReaction.products = ["Water"]) →
  EventCertainty.Certain = 
    match hydrogenOxygenReaction with
    | { reactants := ["Hydrogen", "Oxygen"], products := ["Water"] } => EventCertainty.Certain
    | _ => EventCertainty.Possible
  := by sorry

end hydrogen_oxygen_reaction_certain_l1741_174153


namespace parabola_area_theorem_l1741_174186

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Theorem: For a parabola y^2 = px with p > 0, focus F on the x-axis, and a slanted line through F
    intersecting the parabola at A and B, if the area of triangle OAB is 2√2 (where O is the origin),
    then p = 4√2. -/
theorem parabola_area_theorem (par : Parabola) (F A B : Point) :
  F.x = par.p / 2 →  -- Focus F is on x-axis
  F.y = 0 →
  (∃ m b : ℝ, A.y = m * A.x + b ∧ B.y = m * B.x + b ∧ F.y = m * F.x + b) →  -- A, B, F are on a slanted line
  A.y^2 = par.p * A.x →  -- A is on the parabola
  B.y^2 = par.p * B.x →  -- B is on the parabola
  abs ((A.x * B.y - B.x * A.y) / 2) = 2 * Real.sqrt 2 →  -- Area of triangle OAB is 2√2
  par.p = 4 * Real.sqrt 2 := by
  sorry

end parabola_area_theorem_l1741_174186


namespace perpendicular_lines_b_value_l1741_174162

/-- Given two perpendicular lines with direction vectors (4, -5) and (b, 8), prove that b = 10 -/
theorem perpendicular_lines_b_value :
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ (i j : Fin 2), i ≠ j → v1 i * v2 i + v1 j * v2 j = 0) →
  b = 10 := by
sorry

end perpendicular_lines_b_value_l1741_174162


namespace dodecagon_min_rotation_l1741_174100

/-- The minimum rotation angle for a regular dodecagon to coincide with itself -/
def min_rotation_angle_dodecagon : ℝ := 30

/-- Theorem: The minimum rotation angle for a regular dodecagon to coincide with itself is 30° -/
theorem dodecagon_min_rotation :
  min_rotation_angle_dodecagon = 30 := by sorry

end dodecagon_min_rotation_l1741_174100


namespace f_properties_l1741_174135

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

theorem f_properties :
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max ∧ f x_max = 0) ∧
  (∀ (a : ℝ), a ≥ 2 → ∀ (x : ℝ), x > 0 → f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 → 
    f x₁ + f x₂ + 2 * (x₁^2 + x₂^2) + x₁ * x₂ = 0 → 
    x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end f_properties_l1741_174135


namespace sneakers_cost_l1741_174107

theorem sneakers_cost (sneakers_cost socks_cost : ℝ) 
  (total_cost : sneakers_cost + socks_cost = 101)
  (cost_difference : sneakers_cost = socks_cost + 100) : 
  sneakers_cost = 100.5 := by
  sorry

end sneakers_cost_l1741_174107


namespace product_of_fractions_l1741_174171

theorem product_of_fractions : (1 + 1/3) * (1 + 1/4) = 5/3 := by sorry

end product_of_fractions_l1741_174171


namespace product_of_digits_8056_base_8_l1741_174193

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

theorem product_of_digits_8056_base_8 :
  (base_8_representation 8056).foldl (·*·) 1 = 0 := by
  sorry

end product_of_digits_8056_base_8_l1741_174193


namespace unique_a_value_l1741_174151

/-- A function to represent the exponent |a-2| --/
def abs_a_minus_2 (a : ℝ) : ℝ := |a - 2|

/-- The coefficient of x in the equation --/
def coeff_x (a : ℝ) : ℝ := a - 3

/-- Predicate to check if the equation is linear --/
def is_linear (a : ℝ) : Prop := abs_a_minus_2 a = 1

/-- Theorem stating that a = 1 is the only value satisfying the conditions --/
theorem unique_a_value : ∃! a : ℝ, is_linear a ∧ coeff_x a ≠ 0 := by
  sorry

end unique_a_value_l1741_174151


namespace fraction_to_decimal_l1741_174195

theorem fraction_to_decimal : (11 : ℚ) / 16 = 0.6875 := by
  sorry

end fraction_to_decimal_l1741_174195


namespace cosine_sum_zero_implies_angle_difference_l1741_174170

theorem cosine_sum_zero_implies_angle_difference (α β γ : ℝ) 
  (h1 : 0 < α ∧ α < β ∧ β < γ ∧ γ < 2 * π)
  (h2 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
  γ - α = 4 * π / 3 := by
sorry

end cosine_sum_zero_implies_angle_difference_l1741_174170


namespace bike_race_distance_difference_l1741_174112

theorem bike_race_distance_difference 
  (race_duration : ℝ) 
  (clara_speed : ℝ) 
  (denise_speed : ℝ) 
  (h1 : race_duration = 5)
  (h2 : clara_speed = 18)
  (h3 : denise_speed = 16) :
  clara_speed * race_duration - denise_speed * race_duration = 10 := by
sorry

end bike_race_distance_difference_l1741_174112


namespace roots_of_polynomial_l1741_174199

def f (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -2) := by
sorry

end roots_of_polynomial_l1741_174199


namespace unique_m_value_l1741_174167

def U (m : ℝ) : Set ℝ := {4, m^2 + 2*m - 3, 19}
def A : Set ℝ := {5}

theorem unique_m_value :
  ∃! m : ℝ, (U m \ A = {|4*m - 3|, 4}) ∧ (m^2 + 2*m - 3 = 5) :=
by sorry

end unique_m_value_l1741_174167


namespace quadratic_roots_integrality_l1741_174177

/-- Given two quadratic equations x^2 - px + q = 0 and x^2 - (p+1)x + q = 0,
    this theorem states that when q > 0, both equations can have integer roots,
    but when q < 0, they cannot both have integer roots simultaneously. -/
theorem quadratic_roots_integrality (p q : ℤ) :
  (q > 0 → ∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) ∧
  (q < 0 → ¬∃ (x₁ x₂ x₃ x₄ : ℤ),
    x₁^2 - p*x₁ + q = 0 ∧
    x₂^2 - p*x₂ + q = 0 ∧
    x₃^2 - (p+1)*x₃ + q = 0 ∧
    x₄^2 - (p+1)*x₄ + q = 0) :=
by sorry

end quadratic_roots_integrality_l1741_174177


namespace ray_AB_bisects_angle_PAQ_l1741_174188

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 5/2)^2 = 25/4

-- Define points T, A, and B
def point_T : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (0, 1)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2/8 + y^2/4 = 1

-- Define line l passing through B
def line_l (x y : ℝ) : Prop :=
  ∃ k, y = k * x + 1

-- Define points P and Q as intersections of line l and the ellipse
def point_P : ℝ × ℝ := sorry
def point_Q : ℝ × ℝ := sorry

-- State the theorem
theorem ray_AB_bisects_angle_PAQ :
  circle_C point_T.1 point_T.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 ∧
  point_A.2 > point_B.2 ∧
  point_A.2 - point_B.2 = 3 ∧
  line_l point_P.1 point_P.2 ∧
  line_l point_Q.1 point_Q.2 ∧
  ellipse point_P.1 point_P.2 ∧
  ellipse point_Q.1 point_Q.2 →
  -- The conclusion that ray AB bisects angle PAQ
  -- This would typically involve showing that the angles are equal
  -- or that the dot product of vectors is zero, but we'll leave it as 'sorry'
  sorry :=
sorry

end ray_AB_bisects_angle_PAQ_l1741_174188


namespace division_by_fraction_l1741_174182

theorem division_by_fraction : (10 + 6) / (1 / 4) = 64 := by
  sorry

end division_by_fraction_l1741_174182


namespace fraction_nonzero_digits_l1741_174149

def fraction : ℚ := 120 / (2^4 * 5^8)

def count_nonzero_decimal_digits (q : ℚ) : ℕ :=
  -- Function to count non-zero digits after the decimal point
  sorry

theorem fraction_nonzero_digits :
  count_nonzero_decimal_digits fraction = 3 := by sorry

end fraction_nonzero_digits_l1741_174149


namespace four_leaf_area_l1741_174139

/-- The area of a four-leaf shape formed by semicircles drawn on each side of a square -/
theorem four_leaf_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let leaf_area := π * semicircle_radius^2 / 2 - square_side^2 / 4
  4 * leaf_area = a^2 / 2 * (π - 2) :=
by sorry

end four_leaf_area_l1741_174139


namespace division_problem_l1741_174161

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end division_problem_l1741_174161


namespace x_intercept_of_line_l1741_174104

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l1741_174104


namespace square_side_length_l1741_174138

theorem square_side_length (x : ℝ) (h : x > 0) (h_area : x^2 = 4 * 3) : x = 2 * Real.sqrt 3 := by
  sorry

end square_side_length_l1741_174138


namespace larry_gave_candies_l1741_174166

/-- Given that Anna starts with 5 candies and ends up with 91 candies after receiving some from Larry,
    prove that Larry gave Anna 86 candies. -/
theorem larry_gave_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 5)
  (h2 : final_candies = 91) :
  final_candies - initial_candies = 86 := by
  sorry

end larry_gave_candies_l1741_174166


namespace odometer_reading_l1741_174174

theorem odometer_reading (initial_reading traveled_distance : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : traveled_distance = 159.7) :
  initial_reading + traveled_distance = 372.0 := by
sorry

end odometer_reading_l1741_174174


namespace divisibility_by_30_l1741_174169

theorem divisibility_by_30 (a m n : ℕ) (k : ℤ) 
  (h1 : m > n) (h2 : n ≥ 2) (h3 : m - n = 4 * k.natAbs) : 
  ∃ (q : ℤ), a^m - a^n = 30 * q :=
sorry

end divisibility_by_30_l1741_174169


namespace cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l1741_174143

/-- Pentagon formed by removing a triangle from a rectangle --/
structure CutRectanglePentagon where
  sides : Finset ℕ
  side_count : sides.card = 5
  side_values : sides = {14, 21, 22, 28, 35}

/-- Theorem stating the area of the specific pentagon --/
theorem cut_rectangle_pentagon_area (p : CutRectanglePentagon) : ℕ :=
  1176

#check cut_rectangle_pentagon_area

/-- Proof of the theorem --/
theorem cut_rectangle_pentagon_area_proof (p : CutRectanglePentagon) :
  cut_rectangle_pentagon_area p = 1176 := by
  sorry

end cut_rectangle_pentagon_area_cut_rectangle_pentagon_area_proof_l1741_174143


namespace function_range_l1741_174185

/-- Given a real number m and a function f, prove that if there exists x₀ satisfying certain conditions, then m belongs to the specified range. -/
theorem function_range (m : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = Real.sqrt 3 * Real.sin (π * x / m)) :
  (∃ x₀, (f x₀ = Real.sqrt 3 ∨ f x₀ = -Real.sqrt 3) ∧ x₀^2 + (f x₀)^2 < m^2) →
  m < -2 ∨ m > 2 :=
by sorry

end function_range_l1741_174185


namespace circle_area_sum_l1741_174133

/-- The sum of the areas of an infinite sequence of circles, where the first circle
    has a radius of 3 inches and each subsequent circle's radius is 2/3 of the previous one,
    is equal to 81π/5. -/
theorem circle_area_sum : 
  let r : ℕ → ℝ := λ n => 3 * (2/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (r n)^2
  (∑' n, area n) = 81*π/5 :=
sorry

end circle_area_sum_l1741_174133


namespace flight_duration_sum_l1741_174102

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 →
  departureLA.minutes = 15 →
  arrivalNY.hours = 17 →
  arrivalNY.minutes = 40 →
  0 < m →
  m < 60 →
  timeDiffInMinutes 
    {hours := departureLA.hours + 3, minutes := departureLA.minutes, valid := sorry}
    arrivalNY = h * 60 + m →
  h + m = 30 := by
  sorry

end flight_duration_sum_l1741_174102


namespace wine_glass_ball_radius_l1741_174173

theorem wine_glass_ball_radius 
  (parabola : ℝ → ℝ → Prop) 
  (h_parabola : ∀ x y, parabola x y ↔ x^2 = 2*y) 
  (h_y_range : ∀ y, parabola x y → 0 ≤ y ∧ y ≤ 20) 
  (ball_touches_bottom : ∃ r, r > 0 ∧ ∀ x y, parabola x y → x^2 + y^2 ≥ r^2) :
  ∃ r, r > 0 ∧ r ≤ 1 ∧ 
    (∀ x y, parabola x y → x^2 + y^2 ≥ r^2) ∧
    (∀ r', r' > 0 ∧ r' ≤ 1 → 
      (∀ x y, parabola x y → x^2 + y^2 ≥ r'^2) → 
      r' ≤ r) :=
by sorry

end wine_glass_ball_radius_l1741_174173


namespace second_number_is_thirty_l1741_174117

theorem second_number_is_thirty
  (a b c : ℝ)
  (sum_eq_98 : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  b = 30 := by
  sorry

end second_number_is_thirty_l1741_174117


namespace only_B_on_x_axis_l1741_174146

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis : 
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end only_B_on_x_axis_l1741_174146


namespace quadratic_expression_values_l1741_174128

theorem quadratic_expression_values (a b : ℝ) 
  (ha : a^2 = 16)
  (hb : abs b = 3)
  (hab : a * b < 0) :
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := by
sorry

end quadratic_expression_values_l1741_174128


namespace power_multiplication_l1741_174191

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l1741_174191


namespace books_left_over_l1741_174101

theorem books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ)
  (h1 : initial_boxes = 1575)
  (h2 : books_per_initial_box = 45)
  (h3 : books_per_new_box = 46) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 15 := by
  sorry

end books_left_over_l1741_174101


namespace airbnb_rental_cost_l1741_174136

/-- Calculates the Airbnb rental cost for a vacation -/
theorem airbnb_rental_cost 
  (num_people : ℕ) 
  (car_rental_cost : ℕ) 
  (share_per_person : ℕ) 
  (h1 : num_people = 8)
  (h2 : car_rental_cost = 800)
  (h3 : share_per_person = 500) :
  num_people * share_per_person - car_rental_cost = 3200 := by
  sorry

end airbnb_rental_cost_l1741_174136


namespace lemonade_percentage_l1741_174141

/-- Proves that the percentage of lemonade in the second solution is 45% -/
theorem lemonade_percentage
  (first_solution_carbonated : ℝ)
  (second_solution_carbonated : ℝ)
  (mixture_ratio : ℝ)
  (mixture_carbonated : ℝ)
  (h1 : first_solution_carbonated = 0.8)
  (h2 : second_solution_carbonated = 0.55)
  (h3 : mixture_ratio = 0.5)
  (h4 : mixture_carbonated = 0.675)
  (h5 : mixture_ratio * first_solution_carbonated + (1 - mixture_ratio) * second_solution_carbonated = mixture_carbonated) :
  1 - second_solution_carbonated = 0.45 :=
by sorry

end lemonade_percentage_l1741_174141


namespace hamburgers_leftover_count_l1741_174144

/-- The number of hamburgers made by the restaurant -/
def hamburgers_made : ℕ := 9

/-- The number of hamburgers served during lunch -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := hamburgers_made - hamburgers_served

theorem hamburgers_leftover_count : hamburgers_leftover = 6 := by
  sorry

end hamburgers_leftover_count_l1741_174144


namespace exponential_inequality_l1741_174103

theorem exponential_inequality (m n : ℝ) : (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n → m > n := by
  sorry

end exponential_inequality_l1741_174103


namespace initial_strawberry_plants_l1741_174123

/-- The number of strawberry plants after n months of doubling -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the initial number of strawberry plants -/
theorem initial_strawberry_plants : ∃ (initial : ℕ), 
  plants_after_months initial 3 - 4 = 20 ∧ initial > 0 := by
  sorry

end initial_strawberry_plants_l1741_174123


namespace not_perfect_square_l1741_174129

theorem not_perfect_square : 
  (∃ a : ℕ, 1^2016 = a^2) ∧ 
  (∀ b : ℕ, 2^2017 ≠ b^2) ∧ 
  (∃ c : ℕ, 3^2018 = c^2) ∧ 
  (∃ d : ℕ, 4^2019 = d^2) ∧ 
  (∃ e : ℕ, 5^2020 = e^2) := by
sorry

end not_perfect_square_l1741_174129


namespace gcd_of_polynomial_and_linear_l1741_174118

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1428 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 30)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end gcd_of_polynomial_and_linear_l1741_174118


namespace phone_charges_count_l1741_174160

def daily_mileages : List Nat := [135, 259, 159, 189]
def charge_interval : Nat := 106

theorem phone_charges_count : 
  (daily_mileages.sum / charge_interval : Nat) = 7 := by
  sorry

end phone_charges_count_l1741_174160


namespace min_sum_squared_l1741_174159

theorem min_sum_squared (x₁ x₂ : ℝ) (h : x₁ * x₂ = 2013) : 
  (x₁ + x₂)^2 ≥ 8052 ∧ ∃ y₁ y₂ : ℝ, y₁ * y₂ = 2013 ∧ (y₁ + y₂)^2 = 8052 := by
  sorry

end min_sum_squared_l1741_174159


namespace cafeteria_apple_count_l1741_174148

def initial_apples : ℕ := 65
def used_percentage : ℚ := 20 / 100
def bought_apples : ℕ := 15

theorem cafeteria_apple_count : 
  initial_apples - (initial_apples * used_percentage).floor + bought_apples = 67 :=
by sorry

end cafeteria_apple_count_l1741_174148


namespace circle_center_radius_sum_l1741_174165

theorem circle_center_radius_sum : ∃ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 14*x + y^2 + 6*y = 25 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
  a + b + r = 4 + Real.sqrt 83 := by
  sorry

end circle_center_radius_sum_l1741_174165


namespace negation_of_universal_proposition_l1741_174131

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.log x > 1) ↔ (∃ x : ℝ, Real.log x ≤ 1) := by
  sorry

end negation_of_universal_proposition_l1741_174131


namespace min_value_inequality_l1741_174175

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = Real.sqrt 3) :
  1/a^2 + 2/b^2 ≥ 2 ∧ ∀ x, f x ≥ 2 := by
  sorry

end min_value_inequality_l1741_174175


namespace smallest_k_is_2011_l1741_174109

def is_valid_sequence (s : ℕ → ℕ) : Prop :=
  (∀ n, s n < s (n + 1)) ∧
  (∀ n, (1005 ∣ s n) ∨ (1006 ∣ s n)) ∧
  (∀ n, ¬(97 ∣ s n)) ∧
  (∀ n, s (n + 1) - s n ≤ 2011)

theorem smallest_k_is_2011 :
  ∀ k : ℕ, (∃ s : ℕ → ℕ, is_valid_sequence s ∧ ∀ n, s (n + 1) - s n ≤ k) → k ≥ 2011 :=
sorry

end smallest_k_is_2011_l1741_174109


namespace solve_system_l1741_174126

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 6) : x = 3 := by
  sorry

end solve_system_l1741_174126


namespace fraction_equation_l1741_174127

theorem fraction_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 3) : 
  a / b = 1.2 - 0.4 * Real.sqrt 26 := by
  sorry

end fraction_equation_l1741_174127


namespace max_students_above_average_l1741_174179

theorem max_students_above_average (n : ℕ) (scores : Fin n → ℝ) : 
  n = 80 → (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card) → 
  (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧ k ≤ 79) :=
by sorry

end max_students_above_average_l1741_174179


namespace marathon_remainder_l1741_174176

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
    (h : Distance.mk m y = 
      { miles := num_marathons * marathon.miles + (num_marathons * marathon.yards) / yards_per_mile,
        yards := (num_marathons * marathon.yards) % yards_per_mile }) :
  y = 495 := by sorry

end marathon_remainder_l1741_174176


namespace kyle_stars_theorem_l1741_174184

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) : ℕ :=
  (initial_bottles + additional_bottles) * stars_per_bottle

/-- Theorem stating the total number of stars Kyle needs to make -/
theorem kyle_stars_theorem :
  total_stars 2 3 15 = 75 := by
  sorry

end kyle_stars_theorem_l1741_174184


namespace longest_side_of_triangle_l1741_174106

theorem longest_side_of_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given conditions
  (Real.tan A = 1/4) →
  (Real.tan B = 3/5) →
  (min a (min b c) = Real.sqrt 2) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Conclusion
  max a (max b c) = Real.sqrt 17 :=
by sorry

end longest_side_of_triangle_l1741_174106


namespace median_in_60_64_interval_l1741_174150

/-- Represents a score interval with its frequency -/
structure ScoreInterval :=
  (lowerBound upperBound : ℕ)
  (frequency : ℕ)

/-- The problem setup -/
def testScores : List ScoreInterval :=
  [ ⟨45, 49, 8⟩
  , ⟨50, 54, 15⟩
  , ⟨55, 59, 20⟩
  , ⟨60, 64, 18⟩
  , ⟨65, 69, 17⟩
  , ⟨70, 74, 12⟩
  , ⟨75, 79, 9⟩
  , ⟨80, 84, 6⟩
  ]

def totalStudents : ℕ := 105

/-- The median interval is the one containing the (n+1)/2 th student -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median -/
def findMedianInterval (scores : List ScoreInterval) (medianPos : ℕ) : Option ScoreInterval :=
  let rec go (acc : ℕ) (remaining : List ScoreInterval) : Option ScoreInterval :=
    match remaining with
    | [] => none
    | interval :: rest =>
      let newAcc := acc + interval.frequency
      if newAcc ≥ medianPos then some interval
      else go newAcc rest
  go 0 scores

/-- Theorem stating that the median score is in the interval 60-64 -/
theorem median_in_60_64_interval :
  findMedianInterval testScores medianPosition = some ⟨60, 64, 18⟩ := by
  sorry


end median_in_60_64_interval_l1741_174150


namespace philips_banana_groups_l1741_174111

theorem philips_banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end philips_banana_groups_l1741_174111


namespace number_exceeding_fraction_l1741_174157

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 25 ∧ x = 40 := by
  sorry

end number_exceeding_fraction_l1741_174157


namespace cubic_root_sum_l1741_174194

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  24 * a^3 - 36 * a^2 + 16 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 16 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 16 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end cubic_root_sum_l1741_174194


namespace gomoku_piece_count_l1741_174116

/-- Represents the number of pieces in the Gomoku game box -/
structure GomokuBox where
  initial_black : ℕ
  initial_white : ℕ
  added_black : ℕ
  added_white : ℕ

/-- Theorem statement for the Gomoku piece counting problem -/
theorem gomoku_piece_count (box : GomokuBox) : 
  box.initial_black = box.initial_white ∧ 
  box.initial_black + box.initial_white ≤ 10 ∧
  box.added_black + box.added_white = 20 ∧
  7 * (box.initial_white + box.added_white) = 8 * (box.initial_black + box.added_black) →
  box.initial_black + box.added_black = 16 := by
  sorry

end gomoku_piece_count_l1741_174116


namespace equilateral_triangle_area_perimeter_ratio_l1741_174189

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 is √3/2 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l1741_174189


namespace range_of_a_l1741_174145

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end range_of_a_l1741_174145


namespace expression_equality_l1741_174122

theorem expression_equality (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x - y) * (2 * x + y) = 5 * y^2 := by
  sorry

end expression_equality_l1741_174122


namespace binomial_square_last_term_l1741_174154

theorem binomial_square_last_term (a b : ℝ) :
  ∃ x y : ℝ, x^2 - 10*x*y + 25*y^2 = (x + y)^2 :=
by sorry

end binomial_square_last_term_l1741_174154


namespace emma_sandwich_combinations_l1741_174110

def num_meat : ℕ := 12
def num_cheese : ℕ := 11

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2)

theorem emma_sandwich_combinations :
  sandwich_combinations = 3630 := by sorry

end emma_sandwich_combinations_l1741_174110


namespace sum_of_max_min_values_l1741_174190

theorem sum_of_max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = x + y + z) :
  ∃ (min_val max_val : ℝ),
    (∀ a b c : ℝ, a^2 + b^2 + c^2 = a + b + c → min_val ≤ a + b + c ∧ a + b + c ≤ max_val) ∧
    min_val + max_val = 3 :=
by sorry

end sum_of_max_min_values_l1741_174190


namespace money_division_l1741_174114

theorem money_division (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →
  3 * total / 22 = p →
  7 * total / 22 = q →
  12 * total / 22 = r →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end money_division_l1741_174114


namespace no_solution_to_system_l1741_174172

theorem no_solution_to_system : ¬∃ x : ℝ, (Real.arccos (Real.cos x) = x / 3) ∧ (Real.sin x = Real.cos (x / 3)) := by
  sorry

end no_solution_to_system_l1741_174172


namespace det_scalar_multiple_l1741_174164

theorem det_scalar_multiple {a b c d : ℝ} (h : Matrix.det !![a, b; c, d] = 5) :
  Matrix.det !![3*a, 3*b; 3*c, 3*d] = 45 := by
  sorry

end det_scalar_multiple_l1741_174164


namespace salary_reduction_percentage_l1741_174187

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) :
  (S - R / 100 * S) * (1 + 25 / 100) = S → R = 20 := by
  sorry

end salary_reduction_percentage_l1741_174187


namespace sin_alpha_minus_pi_4_increases_with_k_l1741_174132

theorem sin_alpha_minus_pi_4_increases_with_k (α : Real) (k : Real)
  (h1 : 0 < α)
  (h2 : α < π / 4)
  (h3 : (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = k) :
  ∀ ε > 0, ∃ δ > 0, ∀ k' > k,
    k' - k < δ → Real.sin (α - π / 4) < Real.sin (α - π / 4) + ε :=
by sorry

end sin_alpha_minus_pi_4_increases_with_k_l1741_174132


namespace cookie_jar_remaining_l1741_174113

/-- The amount left in the cookie jar after Doris and Martha's spending -/
theorem cookie_jar_remaining (initial_amount : ℕ) (doris_spent : ℕ) (martha_spent : ℕ) : 
  initial_amount = 21 → 
  doris_spent = 6 → 
  martha_spent = doris_spent / 2 → 
  initial_amount - (doris_spent + martha_spent) = 12 :=
by
  sorry

end cookie_jar_remaining_l1741_174113


namespace unused_cubes_for_5x5x5_with_9_tunnels_l1741_174130

/-- Represents a large cube made of small cubes with tunnels --/
structure LargeCube where
  size : Nat
  numTunnels : Nat

/-- Calculates the number of unused small cubes in a large cube with tunnels --/
def unusedCubes (c : LargeCube) : Nat :=
  c.size^3 - (c.numTunnels * c.size - 6)

/-- Theorem stating that for a 5x5x5 cube with 9 tunnels, 39 small cubes are unused --/
theorem unused_cubes_for_5x5x5_with_9_tunnels :
  let c : LargeCube := { size := 5, numTunnels := 9 }
  unusedCubes c = 39 := by
  sorry

#eval unusedCubes { size := 5, numTunnels := 9 }

end unused_cubes_for_5x5x5_with_9_tunnels_l1741_174130


namespace probability_of_shaded_triangle_l1741_174105

/-- Given a diagram with triangles, this theorem proves the probability of selecting a shaded triangle -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_triangles = 5 → shaded_triangles = 3 → (shaded_triangles : ℚ) / total_triangles = 3 / 5 := by
  sorry

end probability_of_shaded_triangle_l1741_174105


namespace divisors_of_factorial_8_l1741_174119

theorem divisors_of_factorial_8 : (Nat.divisors (Nat.factorial 8)).card = 120 := by
  sorry

end divisors_of_factorial_8_l1741_174119


namespace inequality_system_integer_solutions_l1741_174180

def inequality_system (x : ℝ) : Prop :=
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)

def integer_solutions : Set ℤ :=
  {-1, 0, 1}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), (n ∈ integer_solutions) ↔ (inequality_system (n : ℝ)) :=
sorry

end inequality_system_integer_solutions_l1741_174180


namespace quadratic_two_roots_condition_l1741_174108

theorem quadratic_two_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + c = x₁ + 2 ∧ x₂^2 - 3*x₂ + c = x₂ + 2) ↔ c < 6 := by
  sorry

end quadratic_two_roots_condition_l1741_174108


namespace grinder_loss_percentage_l1741_174198

/-- Represents the financial transaction of buying and selling items --/
structure Transaction where
  grinder_cp : ℝ  -- Cost price of grinder
  mobile_cp : ℝ   -- Cost price of mobile
  mobile_profit_percent : ℝ  -- Profit percentage on mobile
  total_profit : ℝ  -- Overall profit
  grinder_loss_percent : ℝ  -- Loss percentage on grinder (to be proved)

/-- Theorem stating the conditions and the result to be proved --/
theorem grinder_loss_percentage
  (t : Transaction)
  (h1 : t.grinder_cp = 15000)
  (h2 : t.mobile_cp = 8000)
  (h3 : t.mobile_profit_percent = 10)
  (h4 : t.total_profit = 500)
  : t.grinder_loss_percent = 2 := by
  sorry


end grinder_loss_percentage_l1741_174198


namespace cistern_wet_surface_area_l1741_174178

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The total wet surface area of a cistern with given dimensions -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 12 14 1.25 = 233 := by
  sorry

#eval total_wet_surface_area 12 14 1.25

end cistern_wet_surface_area_l1741_174178


namespace three_k_values_with_integer_roots_l1741_174168

/-- A quadratic equation with coefficient k has only integer roots -/
def has_only_integer_roots (k : ℝ) : Prop :=
  ∃ r s : ℤ, ∀ x : ℝ, x^2 + k*x + 4*k = 0 ↔ (x = r ∨ x = s)

/-- The set of real numbers k for which the quadratic equation has only integer roots -/
def integer_root_k_values : Set ℝ :=
  {k : ℝ | has_only_integer_roots k}

/-- There are exactly three values of k for which the quadratic equation has only integer roots -/
theorem three_k_values_with_integer_roots :
  ∃ k₁ k₂ k₃ : ℝ, k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₂ ≠ k₃ ∧
  integer_root_k_values = {k₁, k₂, k₃} :=
sorry

end three_k_values_with_integer_roots_l1741_174168


namespace mo_drink_difference_l1741_174121

theorem mo_drink_difference (n : ℕ) : 
  (n ≥ 0) →  -- n is non-negative
  (2 * n + 5 * 4 = 26) →  -- total cups constraint
  (5 * 4 - 2 * n = 14) :=  -- difference between tea and hot chocolate
by
  sorry

end mo_drink_difference_l1741_174121


namespace main_rectangle_tiled_by_tetraminoes_l1741_174124

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetramino (2 × 3 rectangle with two opposite corners removed) -/
def Tetramino : Rectangle :=
  { width := 2, height := 3 }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a tetramino -/
def tetraminoArea : ℕ :=
  area Tetramino - 2

/-- The main rectangle to be tiled -/
def mainRectangle : Rectangle :=
  { width := 2008, height := 2010 }

/-- Theorem: The main rectangle can be tiled using only tetraminoes -/
theorem main_rectangle_tiled_by_tetraminoes :
  ∃ (n : ℕ), n * tetraminoArea = area mainRectangle :=
sorry

end main_rectangle_tiled_by_tetraminoes_l1741_174124


namespace range_of_3x_minus_y_l1741_174140

theorem range_of_3x_minus_y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 3) : 
  1 ≤ 3*x - y ∧ 3*x - y ≤ 7 := by
sorry

end range_of_3x_minus_y_l1741_174140


namespace isosceles_obtuse_triangle_smallest_angle_l1741_174147

/-- 
Given an isosceles, obtuse triangle where one angle is 75% larger than a right angle,
prove that the measure of each of the two smallest angles is 45/4 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (h_isosceles : α = β)
  (h_obtuse : γ > 90)
  (h_large_angle : γ = 90 * 1.75)
  (h_angle_sum : α + β + γ = 180) : 
  α = 45 / 4 := by
sorry

end isosceles_obtuse_triangle_smallest_angle_l1741_174147


namespace intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1741_174115

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ -6 ≤ a ∧ a ≤ -2 := by sorry

-- Theorem for part (II)
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a < -9 ∨ a > 1 := by sorry

end intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1741_174115


namespace red_ball_probability_not_red_ball_probability_l1741_174120

/-- Represents the set of ball colors in the box -/
inductive BallColor
| Red
| White
| Black

/-- Represents the count of balls for each color -/
def ballCount : BallColor → ℕ
| BallColor.Red => 3
| BallColor.White => 5
| BallColor.Black => 7

/-- The total number of balls in the box -/
def totalBalls : ℕ := ballCount BallColor.Red + ballCount BallColor.White + ballCount BallColor.Black

/-- The probability of drawing a ball of a specific color -/
def drawProbability (color : BallColor) : ℚ :=
  ballCount color / totalBalls

theorem red_ball_probability :
  drawProbability BallColor.Red = 1 / 5 := by sorry

theorem not_red_ball_probability :
  1 - drawProbability BallColor.Red = 4 / 5 := by sorry

end red_ball_probability_not_red_ball_probability_l1741_174120


namespace parallel_vectors_x_value_l1741_174158

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -2 :=
by
  sorry

end parallel_vectors_x_value_l1741_174158


namespace sqrt_three_between_fractions_l1741_174155

theorem sqrt_three_between_fractions (n : ℕ+) :
  ((n + 3 : ℝ) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4 : ℝ) / (n + 1)) → n = 4 := by
  sorry

end sqrt_three_between_fractions_l1741_174155


namespace quadratic_roots_sum_of_squares_l1741_174192

theorem quadratic_roots_sum_of_squares : 
  ∀ (x₁ x₂ : ℝ), (x₁^2 - 3*x₁ - 5 = 0) → (x₂^2 - 3*x₂ - 5 = 0) → (x₁ ≠ x₂) → 
  x₁^2 + x₂^2 = 19 := by
sorry

end quadratic_roots_sum_of_squares_l1741_174192


namespace library_visitors_average_l1741_174125

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 4
  let totalOtherDays : ℕ := 26
  let totalDays : ℕ := 30
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  (totalVisitors : ℚ) / totalDays

theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitors sundayVisitors otherDayVisitors = 276 := by
  sorry

end library_visitors_average_l1741_174125


namespace arrangements_count_l1741_174137

-- Define the number of people and exits
def num_people : ℕ := 5
def num_exits : ℕ := 4

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ := sorry

-- Theorem statement
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end arrangements_count_l1741_174137


namespace attendance_rate_proof_l1741_174196

theorem attendance_rate_proof (total_students : ℕ) (absent_students : ℕ) :
  total_students = 50 →
  absent_students = 2 →
  (((total_students - absent_students) : ℚ) / total_students) * 100 = 96 := by
  sorry

end attendance_rate_proof_l1741_174196


namespace car_journey_theorem_l1741_174197

theorem car_journey_theorem (local_distance : ℝ) (local_speed : ℝ) (highway_speed : ℝ) (average_speed : ℝ) (highway_distance : ℝ) :
  local_distance = 60 ∧
  local_speed = 20 ∧
  highway_speed = 60 ∧
  average_speed = 36 ∧
  average_speed = (local_distance + highway_distance) / (local_distance / local_speed + highway_distance / highway_speed) →
  highway_distance = 120 := by
sorry

end car_journey_theorem_l1741_174197


namespace expression_simplification_expression_evaluation_l1741_174163

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -6*x + 5*y := by
  sorry

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2*x + y) * (2*x - y) - (2*x - 3*y)^2) / (-2*y) = -16 := by
  sorry

end expression_simplification_expression_evaluation_l1741_174163


namespace base_k_conversion_l1741_174183

theorem base_k_conversion (k : ℕ) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end base_k_conversion_l1741_174183


namespace wire_cut_ratio_l1741_174152

/-- Given a wire cut into two pieces of lengths a and b, where piece a forms a rectangle
    with length twice its width and piece b forms a circle, if the areas of the rectangle
    and circle are equal, then a/b = 3/√(2π). -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ x : ℝ, a = 6 * x ∧ 2 * x^2 = π * (b / (2 * π))^2) →
  a / b = 3 / Real.sqrt (2 * π) := by
  sorry

end wire_cut_ratio_l1741_174152


namespace gold_calculation_l1741_174181

-- Define the amount of gold Greg has
def gregs_gold : ℕ := 20

-- Define Katie's gold in terms of Greg's
def katies_gold : ℕ := 4 * gregs_gold

-- Define the total amount of gold
def total_gold : ℕ := gregs_gold + katies_gold

-- Theorem to prove
theorem gold_calculation : total_gold = 100 := by
  sorry

end gold_calculation_l1741_174181


namespace value_added_to_forty_percent_l1741_174156

theorem value_added_to_forty_percent (N : ℝ) (V : ℝ) : 
  N = 100 → 0.4 * N + V = N → V = 60 := by
  sorry

end value_added_to_forty_percent_l1741_174156
