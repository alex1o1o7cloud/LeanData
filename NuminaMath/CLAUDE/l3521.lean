import Mathlib

namespace NUMINAMATH_CALUDE_four_twos_polynomial_property_l3521_352101

/-- A polynomial that takes the value 2 for four different integer inputs -/
def FourTwosPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  P a = 2 ∧ P b = 2 ∧ P c = 2 ∧ P d = 2

theorem four_twos_polynomial_property (P : ℤ → ℤ) 
  (h : FourTwosPolynomial P) :
  ∀ x : ℤ, P x ≠ 1 ∧ P x ≠ 3 ∧ P x ≠ 5 ∧ P x ≠ 7 ∧ P x ≠ 9 :=
sorry

end NUMINAMATH_CALUDE_four_twos_polynomial_property_l3521_352101


namespace NUMINAMATH_CALUDE_donut_theorem_l3521_352163

def donut_problem (initial : ℕ) : ℕ :=
  let after_bill := initial - 2
  let after_secretary := after_bill - 4
  let after_manager := after_secretary - (after_secretary / 10)
  let after_first_coworkers := after_manager - (after_manager / 3)
  after_first_coworkers - (after_first_coworkers / 2)

theorem donut_theorem : donut_problem 50 = 14 := by
  sorry

end NUMINAMATH_CALUDE_donut_theorem_l3521_352163


namespace NUMINAMATH_CALUDE_soccer_ball_cost_is_6_l3521_352186

/-- The cost of a soccer ball purchased by four friends -/
def soccer_ball_cost (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 = 2.30 ∧
  x2 = (1/3) * (x1 + x3 + x4) ∧
  x3 = (1/4) * (x1 + x2 + x4) ∧
  x4 = (1/5) * (x1 + x2 + x3) ∧
  x1 + x2 + x3 + x4 = 6

theorem soccer_ball_cost_is_6 :
  ∃ x1 x2 x3 x4 : ℝ, soccer_ball_cost x1 x2 x3 x4 :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_is_6_l3521_352186


namespace NUMINAMATH_CALUDE_shopkeeper_profit_loss_l3521_352110

theorem shopkeeper_profit_loss (cost : ℝ) : 
  cost > 0 →
  let profit_percent := 10
  let loss_percent := 10
  let selling_price1 := cost * (1 + profit_percent / 100)
  let selling_price2 := cost * (1 - loss_percent / 100)
  let total_cost := 2 * cost
  let total_selling_price := selling_price1 + selling_price2
  (total_selling_price - total_cost) / total_cost * 100 = 0 :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_loss_l3521_352110


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3521_352188

theorem collinear_points_b_value :
  ∀ b : ℚ,
  (∃ (m c : ℚ), 
    (m * 4 + c = -6) ∧
    (m * (b + 3) + c = -1) ∧
    (m * (-3 * b + 4) + c = 5)) →
  b = 11 / 26 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3521_352188


namespace NUMINAMATH_CALUDE_base_conversion_and_division_l3521_352152

/-- Given that 746 in base 8 is equal to 4cd in base 10, where c and d are base-10 digits,
    prove that (c * d) / 12 = 4 -/
theorem base_conversion_and_division (c d : ℕ) : 
  c < 10 → d < 10 → 746 = 4 * c * 10 + d → (c * d) / 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_division_l3521_352152


namespace NUMINAMATH_CALUDE_first_car_distance_l3521_352114

theorem first_car_distance (total_distance : ℝ) (second_car_distance : ℝ) (side_distance : ℝ) (final_distance : ℝ) 
  (h1 : total_distance = 113)
  (h2 : second_car_distance = 35)
  (h3 : side_distance = 15)
  (h4 : final_distance = 28) :
  ∃ x : ℝ, x = 17.5 ∧ total_distance - (2 * x + side_distance + second_car_distance) = final_distance :=
by
  sorry


end NUMINAMATH_CALUDE_first_car_distance_l3521_352114


namespace NUMINAMATH_CALUDE_power_of_power_three_l3521_352154

theorem power_of_power_three : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3521_352154


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l3521_352116

theorem equidistant_point_on_y_axis : ∃ y : ℚ, 
  let A : ℚ × ℚ := (-3, 1)
  let B : ℚ × ℚ := (-2, 5)
  let P : ℚ × ℚ := (0, y)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ y = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l3521_352116


namespace NUMINAMATH_CALUDE_marias_test_score_l3521_352137

theorem marias_test_score (scores : Fin 4 → ℕ) : 
  scores 0 = 80 →
  scores 2 = 90 →
  scores 3 = 100 →
  (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 85 →
  scores 1 = 70 := by
sorry

end NUMINAMATH_CALUDE_marias_test_score_l3521_352137


namespace NUMINAMATH_CALUDE_smallest_x_divisible_l3521_352119

theorem smallest_x_divisible : ∃ (x : ℤ), x = 36629 ∧ 
  (∀ (y : ℤ), y < x → ¬(33 ∣ (2 * y + 2) ∧ 44 ∣ (2 * y + 2) ∧ 55 ∣ (2 * y + 2) ∧ 666 ∣ (2 * y + 2))) ∧
  (33 ∣ (2 * x + 2) ∧ 44 ∣ (2 * x + 2) ∧ 55 ∣ (2 * x + 2) ∧ 666 ∣ (2 * x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_divisible_l3521_352119


namespace NUMINAMATH_CALUDE_lisa_book_purchase_l3521_352121

theorem lisa_book_purchase (total_volumes : ℕ) (standard_cost deluxe_cost total_cost : ℕ) 
  (h1 : total_volumes = 15)
  (h2 : standard_cost = 20)
  (h3 : deluxe_cost = 30)
  (h4 : total_cost = 390) :
  ∃ (deluxe_count : ℕ), 
    deluxe_count * deluxe_cost + (total_volumes - deluxe_count) * standard_cost = total_cost ∧
    deluxe_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_lisa_book_purchase_l3521_352121


namespace NUMINAMATH_CALUDE_four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l3521_352106

-- Define a quadrilateral
structure Quadrilateral :=
  (is_plane : Bool)
  (right_angles : Nat)

-- Define the property of being a plane figure
def is_plane_figure (q : Quadrilateral) : Prop :=
  q.is_plane = true

-- Define the property of having four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  q.right_angles = 4

-- Theorem stating that a quadrilateral with four right angles must be a plane figure
theorem four_right_angles_implies_plane_figure (q : Quadrilateral) :
  has_four_right_angles q → is_plane_figure q :=
by sorry

-- Theorem stating that quadrilaterals with less than four right angles can be non-planar
theorem less_than_four_right_angles_can_be_non_planar :
  ∃ (q : Quadrilateral), q.right_angles < 4 ∧ ¬(is_plane_figure q) :=
by sorry

end NUMINAMATH_CALUDE_four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l3521_352106


namespace NUMINAMATH_CALUDE_concert_tickets_sold_l3521_352158

theorem concert_tickets_sold (cost_A cost_B total_tickets total_revenue : ℚ)
  (h1 : cost_A = 8)
  (h2 : cost_B = 4.25)
  (h3 : total_tickets = 4500)
  (h4 : total_revenue = 30000)
  : ∃ (tickets_A tickets_B : ℚ),
    tickets_A + tickets_B = total_tickets ∧
    cost_A * tickets_A + cost_B * tickets_B = total_revenue ∧
    tickets_A = 2900 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_sold_l3521_352158


namespace NUMINAMATH_CALUDE_car_ac_price_ratio_l3521_352179

/-- Given a car that costs $500 more than an AC, and the AC costs $1500,
    prove that the ratio of the car's price to the AC's price is 4:3. -/
theorem car_ac_price_ratio :
  ∀ (car_price ac_price : ℕ),
  ac_price = 1500 →
  car_price = ac_price + 500 →
  (car_price : ℚ) / ac_price = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_car_ac_price_ratio_l3521_352179


namespace NUMINAMATH_CALUDE_parabola_vertex_l3521_352182

/-- The vertex of the parabola y = 2(x-3)^2 + 1 is at the point (3, 1). -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x - 3)^2 + 1 → (3, 1) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3521_352182


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3521_352139

theorem sqrt_equation_solution : ∃ (x : ℝ), x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3521_352139


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_to_directrix_l3521_352196

/-- Given a parabola y² = 4x and a chord AB of length 7 intersecting the parabola at points A(x₁, y₁) and B(x₂, y₂),
    the distance from the midpoint M of the chord to the parabola's directrix is 7/2. -/
theorem parabola_chord_midpoint_to_directrix
  (x₁ y₁ x₂ y₂ : ℝ) 
  (on_parabola_A : y₁^2 = 4*x₁)
  (on_parabola_B : y₂^2 = 4*x₂)
  (chord_length : Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7) :
  let midpoint_x := (x₁ + x₂) / 2
  (midpoint_x + 1) = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_to_directrix_l3521_352196


namespace NUMINAMATH_CALUDE_hawkeye_battery_charge_cost_l3521_352134

theorem hawkeye_battery_charge_cost 
  (budget : ℝ) 
  (num_charges : ℕ) 
  (remaining : ℝ) 
  (h1 : budget = 20)
  (h2 : num_charges = 4)
  (h3 : remaining = 6) : 
  (budget - remaining) / num_charges = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_charge_cost_l3521_352134


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3521_352113

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 ∧
  ((x^(k+2) / (x^(k+1) + y^k + z^k)) + 
   (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
   (z^(k+2) / (z^(k+1) + x^k + y^k)) = 1/7 ↔ 
   x = 1/3 ∧ y = 1/3 ∧ z = 1/3) := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3521_352113


namespace NUMINAMATH_CALUDE_unique_aabb_perfect_square_l3521_352120

/-- A 4-digit number of the form aabb in base 10 -/
def aabb (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * b + b

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem unique_aabb_perfect_square :
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    (is_perfect_square (aabb a b) ↔ a = 7 ∧ b = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_aabb_perfect_square_l3521_352120


namespace NUMINAMATH_CALUDE_hyperbola_with_foci_on_y_axis_l3521_352112

theorem hyperbola_with_foci_on_y_axis 
  (m n : ℝ) 
  (h : m * n < 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
    (∀ (c : ℝ), c > a → ∃ (f₁ f₂ : ℝ), 
      f₁ = 0 ∧ f₂ = 0 ∧ 
      ∀ (x y : ℝ), m * x^2 - m * y^2 = n → 
        (x - f₁)^2 + (y - f₂)^2 - ((x - f₁)^2 + (y + f₂)^2) = 4 * c^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_with_foci_on_y_axis_l3521_352112


namespace NUMINAMATH_CALUDE_excursion_dates_correct_l3521_352123

/-- Represents the four excursion locations --/
inductive Location
| Carpathians
| Kyiv
| Forest
| Museum

/-- Represents a calendar month --/
structure Month where
  number : Nat
  days : Nat
  first_day_sunday : Bool

/-- Represents an excursion --/
structure Excursion where
  location : Location
  month : Month
  day : Nat

/-- Checks if a given day is the first Sunday after the first Saturday --/
def is_first_sunday_after_saturday (m : Month) (d : Nat) : Prop :=
  d = 8 ∧ m.first_day_sunday

/-- The theorem to prove --/
theorem excursion_dates_correct (feb mar : Month) 
  (e1 e2 e3 e4 : Excursion) : 
  feb.number = 2 → 
  mar.number = 3 → 
  feb.days = 28 → 
  mar.days = 31 → 
  feb.first_day_sunday = true → 
  mar.first_day_sunday = true → 
  e1.location = Location.Carpathians → 
  e2.location = Location.Kyiv → 
  e3.location = Location.Forest → 
  e4.location = Location.Museum → 
  e1.month = feb ∧ e1.day = 1 ∧
  e2.month = feb ∧ is_first_sunday_after_saturday feb e2.day ∧
  e3.month = mar ∧ e3.day = 1 ∧
  e4.month = mar ∧ is_first_sunday_after_saturday mar e4.day :=
sorry

end NUMINAMATH_CALUDE_excursion_dates_correct_l3521_352123


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3521_352136

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3521_352136


namespace NUMINAMATH_CALUDE_two_congruent_rectangles_l3521_352126

/-- A point on a circle --/
structure CirclePoint where
  angle : ℝ
  angleInRange : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- A rectangle inscribed in a circle --/
structure InscribedRectangle where
  vertices : Fin 4 → CirclePoint
  isRectangle : ∀ i : Fin 4, (vertices i).angle - (vertices ((i + 1) % 4)).angle = Real.pi / 2 ∨
                             (vertices i).angle - (vertices ((i + 1) % 4)).angle = -3 * Real.pi / 2

/-- The main theorem --/
theorem two_congruent_rectangles 
  (points : Fin 40 → CirclePoint)
  (equallySpaced : ∀ i : Fin 39, (points (i + 1)).angle - (points i).angle = Real.pi / 20)
  (rectangles : Fin 10 → InscribedRectangle)
  (verticesOnPoints : ∀ r : Fin 10, ∀ v : Fin 4, ∃ p : Fin 40, (rectangles r).vertices v = points p) :
  ∃ r1 r2 : Fin 10, r1 ≠ r2 ∧ rectangles r1 = rectangles r2 :=
sorry

end NUMINAMATH_CALUDE_two_congruent_rectangles_l3521_352126


namespace NUMINAMATH_CALUDE_ninety_nine_times_one_hundred_one_l3521_352189

theorem ninety_nine_times_one_hundred_one : 99 * 101 = 9999 := by
  sorry

end NUMINAMATH_CALUDE_ninety_nine_times_one_hundred_one_l3521_352189


namespace NUMINAMATH_CALUDE_special_integer_pairs_l3521_352168

theorem special_integer_pairs (a b : ℕ+) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ a^2 + b + 1 = p^k) →
  (a^2 + b + 1) ∣ (b^2 - a^3 - 1) →
  ¬((a^2 + b + 1) ∣ (a + b - 1)^2) →
  ∃ (s : ℕ), s ≥ 2 ∧ a = 2^s ∧ b = 2^(2*s) - 1 :=
by sorry

end NUMINAMATH_CALUDE_special_integer_pairs_l3521_352168


namespace NUMINAMATH_CALUDE_tan_sum_identity_l3521_352199

theorem tan_sum_identity (A B : Real) (hA : A = 10 * π / 180) (hB : B = 20 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 1 + Real.sqrt 3 * (Real.tan A + Real.tan B) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l3521_352199


namespace NUMINAMATH_CALUDE_inequalities_always_true_l3521_352125

theorem inequalities_always_true (a b : ℝ) (h : a * b > 0) :
  (a^2 + b^2 ≥ 2*a*b) ∧ (b/a + a/b ≥ 2) := by sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l3521_352125


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3521_352138

theorem trigonometric_identity (A B C : Real) (h : A + B + C = π) :
  Real.sin A * Real.cos B * Real.cos C + 
  Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = 
  Real.sin A * Real.sin B * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3521_352138


namespace NUMINAMATH_CALUDE_spherical_coordinate_conversion_l3521_352127

theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ = 5 ∧ θ = 5 * Real.pi / 7 ∧ φ = 11 * Real.pi / 6 →
  ∃ (ρ' θ' φ' : Real),
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi ∧
    ρ' = 5 ∧
    θ' = 12 * Real.pi / 7 ∧
    φ' = Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_conversion_l3521_352127


namespace NUMINAMATH_CALUDE_golden_retriever_pup_difference_l3521_352146

/-- Represents the number of pups each golden retriever had more than each husky -/
def pup_difference : ℕ := 2

theorem golden_retriever_pup_difference :
  let num_huskies : ℕ := 5
  let num_pitbulls : ℕ := 2
  let num_golden_retrievers : ℕ := 4
  let pups_per_husky : ℕ := 3
  let pups_per_pitbull : ℕ := 3
  let total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers
  let total_pups : ℕ := total_adult_dogs + 30
  total_pups = num_huskies * pups_per_husky + num_pitbulls * pups_per_pitbull + 
               num_golden_retrievers * (pups_per_husky + pup_difference) →
  pup_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_pup_difference_l3521_352146


namespace NUMINAMATH_CALUDE_gothic_window_radius_l3521_352162

/-- Gothic Window Configuration -/
structure GothicWindow where
  -- O is the center of the circle passing through A and B
  -- P is the center of the inscribed circle
  -- Q is a point on arc BC
  OA : ℝ -- Distance from O to A
  AP : ℝ -- Distance from A to P
  AQ : ℝ -- Distance from A to Q
  r : ℝ  -- Radius of the inscribed circle

/-- Properties of the Gothic Window -/
def gothic_window_properties (w : GothicWindow) : Prop :=
  w.OA = 2 ∧ w.AQ = 4 ∧ w.AP^2 = w.r^2 + 2*w.r + 4

/-- Theorem: The radius of the inscribed circle in the Gothic Window is 6/5 -/
theorem gothic_window_radius (w : GothicWindow) 
  (h : gothic_window_properties w) : w.r = 6/5 := by
  sorry

#check gothic_window_radius

end NUMINAMATH_CALUDE_gothic_window_radius_l3521_352162


namespace NUMINAMATH_CALUDE_polynomial_roots_l3521_352103

/-- The polynomial x^3 + x^2 - 4x - 2 --/
def f (x : ℂ) : ℂ := x^3 + x^2 - 4*x - 2

/-- The roots of the polynomial --/
def roots : List ℂ := [1, -1 + Complex.I, -1 - Complex.I]

theorem polynomial_roots :
  ∀ r ∈ roots, f r = 0 ∧ (∀ z : ℂ, f z = 0 → z ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3521_352103


namespace NUMINAMATH_CALUDE_sin45_plus_sqrt2_half_l3521_352147

theorem sin45_plus_sqrt2_half (h : Real.sin (π / 4) = Real.sqrt 2 / 2) :
  Real.sin (π / 4) + Real.sqrt 2 / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin45_plus_sqrt2_half_l3521_352147


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l3521_352122

theorem mayoral_election_votes 
  (votes_Z : ℕ) 
  (h1 : votes_Z = 25000)
  (votes_Y : ℕ) 
  (h2 : votes_Y = votes_Z - (2 / 5 : ℚ) * votes_Z)
  (votes_X : ℕ) 
  (h3 : votes_X = votes_Y + (1 / 2 : ℚ) * votes_Y) :
  votes_X = 22500 := by
sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l3521_352122


namespace NUMINAMATH_CALUDE_tenth_term_is_negative_eight_l3521_352140

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 7 = 2
  product_condition : a 5 * a 6 = -8
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q

/-- The 10th term of the geometric sequence is -8 -/
theorem tenth_term_is_negative_eight (seq : GeometricSequence) : seq.a 10 = -8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_negative_eight_l3521_352140


namespace NUMINAMATH_CALUDE_rational_product_sum_implies_negative_l3521_352191

theorem rational_product_sum_implies_negative (a b : ℚ) 
  (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_product_sum_implies_negative_l3521_352191


namespace NUMINAMATH_CALUDE_square_root_difference_l3521_352166

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def twos (n : ℕ) : ℕ := 2 * ones n

theorem square_root_difference (n : ℕ+) :
  (ones (2*n) - twos n).sqrt = ones (2*n - 1) * 3 :=
sorry

end NUMINAMATH_CALUDE_square_root_difference_l3521_352166


namespace NUMINAMATH_CALUDE_existence_of_n_for_prime_divisibility_l3521_352150

theorem existence_of_n_for_prime_divisibility (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_for_prime_divisibility_l3521_352150


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3521_352174

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (p : Point)
  (l1 l2 : Line)
  (h1 : p.liesOn l2)
  (h2 : l2.isParallelTo l1)
  (h3 : l1.a = 1)
  (h4 : l1.b = -2)
  (h5 : l1.c = 3)
  (h6 : p.x = 1)
  (h7 : p.y = -3)
  (h8 : l2.a = 1)
  (h9 : l2.b = -2)
  (h10 : l2.c = -7) :
  True := by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3521_352174


namespace NUMINAMATH_CALUDE_total_sibling_age_l3521_352131

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.susan = 15 → 
  ages.bob = 11 → 
  ages.arthur = ages.susan + 2 → 
  ages.tom = ages.bob - 3 → 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_sibling_age_l3521_352131


namespace NUMINAMATH_CALUDE_system_of_equations_substitution_l3521_352165

theorem system_of_equations_substitution :
  ∀ x y : ℝ,
  (2 * x - 5 * y = 4) →
  (3 * x - y = 1) →
  (2 * x - 5 * (3 * x - 1) = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_substitution_l3521_352165


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_chord_length_l3521_352198

/-- Given a hyperbola and a circle, prove the length of the chord formed by their intersection -/
theorem hyperbola_circle_intersection_chord_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola : ℝ → ℝ → Prop) 
  (asymptote : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop) :
  (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
  (asymptote 1 2) →
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  ∃ chord_length, chord_length = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_chord_length_l3521_352198


namespace NUMINAMATH_CALUDE_five_primes_in_valid_set_l3521_352194

/-- The set of digits to choose from -/
def digit_set : Finset Nat := {3, 5, 7, 8}

/-- Function to form a two-digit number from two digits -/
def form_number (tens units : Nat) : Nat := 10 * tens + units

/-- Predicate to check if a number is formed from two different digits in the set -/
def is_valid_number (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ digit_set ∧ units ∈ digit_set ∧ tens ≠ units ∧ n = form_number tens units

/-- The set of all valid two-digit numbers formed from the digit set -/
def valid_numbers : Finset Nat := sorry

/-- The theorem stating that there are exactly 5 prime numbers in the valid set -/
theorem five_primes_in_valid_set : (valid_numbers.filter Nat.Prime).card = 5 := by sorry

end NUMINAMATH_CALUDE_five_primes_in_valid_set_l3521_352194


namespace NUMINAMATH_CALUDE_lines_are_parallel_l3521_352190

/-- Two lines are parallel if they have the same slope -/
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂

/-- The first line: y = -2x + 1 -/
def line1 (x : ℝ) : ℝ := -2 * x + 1

/-- The second line: y = -2x + 3 -/
def line2 (x : ℝ) : ℝ := -2 * x + 3

/-- Theorem: The line y = -2x + 1 is parallel to the line y = -2x + 3 -/
theorem lines_are_parallel : parallel (-2) 1 (-2) 3 := by sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l3521_352190


namespace NUMINAMATH_CALUDE_community_size_after_five_years_l3521_352159

def community_growth (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | m + 1 => 4 * community_growth m - 15

theorem community_size_after_five_years :
  community_growth 5 = 15365 := by
  sorry

end NUMINAMATH_CALUDE_community_size_after_five_years_l3521_352159


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l3521_352197

def M : ℕ := 36 * 36 * 98 * 210

-- Sum of odd divisors
def sum_odd_divisors (n : ℕ) : ℕ := sorry

-- Sum of even divisors
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 60 = sum_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l3521_352197


namespace NUMINAMATH_CALUDE_equation_solution_l3521_352153

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -4 ∧ 
  (∀ x : ℝ, (x - 1) * (x + 3) = 5 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3521_352153


namespace NUMINAMATH_CALUDE_range_of_a_l3521_352156

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x + 2
  else x + a/x + 3*a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3521_352156


namespace NUMINAMATH_CALUDE_seth_sold_78_candy_bars_l3521_352175

def max_candy_bars : ℕ := 24

def seth_candy_bars : ℕ := 3 * max_candy_bars + 6

theorem seth_sold_78_candy_bars : seth_candy_bars = 78 := by
  sorry

end NUMINAMATH_CALUDE_seth_sold_78_candy_bars_l3521_352175


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l3521_352124

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l3521_352124


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l3521_352105

/-- Represents the problem of determining the number of metres of cloth sold --/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 95 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 := by
  sorry

#check cloth_sale_problem

end NUMINAMATH_CALUDE_cloth_sale_problem_l3521_352105


namespace NUMINAMATH_CALUDE_sum_digits_of_numeric_hex_count_l3521_352184

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number --/
def Hexadecimal := List HexDigit

/-- Converts a natural number to hexadecimal --/
def toHex (n : ℕ) : Hexadecimal :=
  sorry

/-- Checks if a hexadecimal number uses only numeric digits --/
def usesOnlyNumericDigits (h : Hexadecimal) : Bool :=
  sorry

/-- Counts numbers representable in hexadecimal using only numeric digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem sum_digits_of_numeric_hex_count :
  sumDigits (countNumericHex 2000) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_of_numeric_hex_count_l3521_352184


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3521_352118

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3521_352118


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3521_352111

/-- Given an ellipse equation (x²/(10-m)) + (y²/(m-2)) = 1 with focal length 4,
    prove that the possible values of m are 4 and 8. -/
theorem ellipse_focal_length (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m)) + (y^2 / (m - 2)) = 1) →
  (∃ a b c : ℝ, a^2 = 10 - m ∧ b^2 = m - 2 ∧ c = 4 ∧ a^2 - b^2 = c^2) →
  m = 4 ∨ m = 8 := by
sorry


end NUMINAMATH_CALUDE_ellipse_focal_length_l3521_352111


namespace NUMINAMATH_CALUDE_most_likely_car_count_l3521_352141

/-- Represents the number of cars counted in a given time interval -/
structure CarCount where
  cars : ℕ
  seconds : ℕ

/-- Represents the total time taken by the train to pass -/
structure TotalTime where
  minutes : ℕ
  seconds : ℕ

/-- Calculates the most likely number of cars in the train -/
def calculateTotalCars (initial_count : CarCount) (total_time : TotalTime) : ℕ :=
  let total_seconds := total_time.minutes * 60 + total_time.seconds
  let rate := initial_count.cars / initial_count.seconds
  rate * total_seconds

/-- Theorem stating that given the conditions, the most likely number of cars is 70 -/
theorem most_likely_car_count 
  (initial_count : CarCount)
  (total_time : TotalTime)
  (h1 : initial_count = ⟨5, 15⟩)
  (h2 : total_time = ⟨3, 30⟩) :
  calculateTotalCars initial_count total_time = 70 := by
  sorry

#eval calculateTotalCars ⟨5, 15⟩ ⟨3, 30⟩

end NUMINAMATH_CALUDE_most_likely_car_count_l3521_352141


namespace NUMINAMATH_CALUDE_last_card_identifiable_determine_last_card_back_l3521_352109

/-- Represents a card with two sides -/
structure Card where
  front : ℕ
  back : ℕ

/-- Creates a deck of n cards -/
def create_deck (n : ℕ) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Checks if a number appears in a list -/
def appears_in (k : ℕ) (list : List ℕ) : Prop :=
  k ∈ list

/-- Theorem: Determine if the back of the last card can be identified -/
theorem last_card_identifiable (n : ℕ) (shown : List ℕ) (last : ℕ) : Prop :=
  let deck := create_deck n
  last = 0 ∨ last = n ∨
  (1 ≤ last ∧ last ≤ n - 1 ∧ (appears_in (last - 1) shown ∨ appears_in (last + 1) shown))

/-- Main theorem: Characterization of when the back of the last card can be determined -/
theorem determine_last_card_back (n : ℕ) (shown : List ℕ) (last : ℕ) :
  last_card_identifiable n shown last ↔
  ∃ (card : Card), card ∈ create_deck n ∧
    ((card.front = last ∧ ∃ k, k ∈ shown ∧ k = card.back) ∨
     (card.back = last ∧ ∃ k, k ∈ shown ∧ k = card.front)) :=
  sorry


end NUMINAMATH_CALUDE_last_card_identifiable_determine_last_card_back_l3521_352109


namespace NUMINAMATH_CALUDE_polynomial_expansions_l3521_352193

theorem polynomial_expansions (x y : ℝ) : 
  ((x - 3) * (x^2 + 4) = x^3 - 3*x^2 + 4*x - 12) ∧ 
  ((3*x^2 - y) * (x + 2*y) = 3*x^3 + 6*y*x^2 - x*y - 2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansions_l3521_352193


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_minus_one_l3521_352104

theorem sum_of_reciprocals_of_roots_minus_one (p q r : ℂ) : 
  (p^3 - p - 2 = 0) → (q^3 - q - 2 = 0) → (r^3 - r - 2 = 0) →
  (1 / (p - 1) + 1 / (q - 1) + 1 / (r - 1) = -2) := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_minus_one_l3521_352104


namespace NUMINAMATH_CALUDE_bread_price_for_cash_register_l3521_352102

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadSold : ℕ
  cakesPrice : ℝ
  cakesSold : ℕ
  rentCost : ℝ
  electricityCost : ℝ

/-- Calculates the daily profit of the bakery --/
def dailyProfit (b : BakeryFinances) : ℝ :=
  b.breadPrice * b.breadSold + b.cakesPrice * b.cakesSold - b.rentCost - b.electricityCost

/-- The main theorem: The price of bread that allows Marie to buy the cash register in 8 days is $2 --/
theorem bread_price_for_cash_register (b : BakeryFinances) 
    (h1 : b.breadSold = 40)
    (h2 : b.cakesSold = 6)
    (h3 : b.cakesPrice = 12)
    (h4 : b.rentCost = 20)
    (h5 : b.electricityCost = 2)
    (h6 : 8 * dailyProfit b = 1040) : 
  b.breadPrice = 2 := by
  sorry

#check bread_price_for_cash_register

end NUMINAMATH_CALUDE_bread_price_for_cash_register_l3521_352102


namespace NUMINAMATH_CALUDE_doug_initial_marbles_l3521_352176

theorem doug_initial_marbles (ed_marbles : ℕ) (ed_more_than_doug : ℕ) (doug_lost : ℕ)
  (h1 : ed_marbles = 27)
  (h2 : ed_more_than_doug = 5)
  (h3 : doug_lost = 3) :
  ed_marbles - ed_more_than_doug + doug_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_doug_initial_marbles_l3521_352176


namespace NUMINAMATH_CALUDE_juice_bread_price_ratio_l3521_352187

/-- Calculates the ratio of juice price to bread price given shopping details --/
theorem juice_bread_price_ratio 
  (total_money : ℝ)
  (bread_price : ℝ)
  (butter_price : ℝ)
  (money_left : ℝ)
  (h1 : total_money = 15)
  (h2 : bread_price = 2)
  (h3 : butter_price = 3)
  (h4 : money_left = 6) :
  (total_money - money_left - bread_price - butter_price) / bread_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_juice_bread_price_ratio_l3521_352187


namespace NUMINAMATH_CALUDE_min_value_of_function_l3521_352180

theorem min_value_of_function (x a b : ℝ) 
  (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) : 
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3521_352180


namespace NUMINAMATH_CALUDE_lorenzo_board_test_l3521_352195

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The total number of thumbtacks Lorenzo started with -/
def total_thumbtacks : ℕ := 450

/-- The number of cans of thumbtacks -/
def number_of_cans : ℕ := 3

/-- The number of thumbtacks remaining in each can at the end of the day -/
def remaining_thumbtacks_per_can : ℕ := 30

/-- The number of thumbtacks used per board -/
def thumbtacks_per_board : ℕ := 3

theorem lorenzo_board_test :
  boards_tested = (total_thumbtacks - number_of_cans * remaining_thumbtacks_per_can) / thumbtacks_per_board :=
by sorry

end NUMINAMATH_CALUDE_lorenzo_board_test_l3521_352195


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3521_352164

/-- Converts a binary number represented as a list of digits to its decimal equivalent -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [1, 1, 1, 0]
  let ternary_num := [1, 0, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 154 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l3521_352164


namespace NUMINAMATH_CALUDE_gain_percent_for_50_and_28_l3521_352161

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles : ℕ) (sellArticles : ℕ) : ℚ :=
  let gain := (costArticles : ℚ) / sellArticles - 1
  gain * 100

/-- Proves that when 50 articles at cost price equal 28 articles at selling price, the gain percent is (11/14) * 100 -/
theorem gain_percent_for_50_and_28 :
  gainPercent 50 28 = 11 / 14 * 100 := by
  sorry

#eval gainPercent 50 28

end NUMINAMATH_CALUDE_gain_percent_for_50_and_28_l3521_352161


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l3521_352183

theorem sqrt_sum_reciprocal (x : ℝ) (hx_pos : x > 0) (hx_sum : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l3521_352183


namespace NUMINAMATH_CALUDE_season_games_count_l3521_352133

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l3521_352133


namespace NUMINAMATH_CALUDE_wendy_sold_nine_pastries_l3521_352130

/-- Represents the number of pastries sold at a bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proves that Wendy sold 9 pastries at the bake sale -/
theorem wendy_sold_nine_pastries :
  pastries_sold 4 29 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_wendy_sold_nine_pastries_l3521_352130


namespace NUMINAMATH_CALUDE_four_integers_average_l3521_352167

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 5 →
    (d - a : ℤ) ≥ (z - w : ℤ) →
  ((b : ℚ) + c) / 2 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_four_integers_average_l3521_352167


namespace NUMINAMATH_CALUDE_binary_11011_to_decimal_l3521_352181

def binary_to_decimal (b₄ b₃ b₂ b₁ b₀ : Nat) : Nat :=
  b₄ * 2^4 + b₃ * 2^3 + b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_11011_to_decimal :
  binary_to_decimal 1 1 0 1 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_to_decimal_l3521_352181


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3521_352144

open Real

theorem angle_sum_is_pi_over_two (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin α ^ 2 + sin β ^ 2 - (Real.sqrt 6 / 2) * sin α - (Real.sqrt 10 / 2) * sin β + 1 = 0) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3521_352144


namespace NUMINAMATH_CALUDE_distance_between_locations_l3521_352160

/-- The distance between two locations A and B given the conditions of two couriers --/
theorem distance_between_locations (x : ℝ) (y : ℝ) : 
  (x > 0) →  -- x is the number of days until the couriers meet
  (y > 0) →  -- y is the total distance between A and B
  (y / (x + 9) + y / (x + 16) = y) →  -- sum of distances traveled equals total distance
  (y / (x + 9) - y / (x + 16) = 12) →  -- difference in distances traveled is 12 miles
  (x^2 = 144) →  -- derived from solving the equations
  y = 84 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l3521_352160


namespace NUMINAMATH_CALUDE_sculpture_cost_in_pesos_l3521_352115

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Mexican pesos -/
def usd_to_mxn : ℝ := 20

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating that the cost of the sculpture in Mexican pesos is 400 -/
theorem sculpture_cost_in_pesos :
  (sculpture_cost_nad / usd_to_nad) * usd_to_mxn = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_pesos_l3521_352115


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3521_352155

/-- If the solution set of (1-m^2)x^2-(1+m)x-1<0 with respect to x is ℝ,
    then m satisfies m ≤ -1 or m > 5/3 -/
theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) →
  (m ≤ -1 ∨ m > 5/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3521_352155


namespace NUMINAMATH_CALUDE_goldfish_equality_l3521_352145

theorem goldfish_equality (n : ℕ) : (∃ m : ℕ, m < n ∧ 3^(m + 1) = 81 * 3^m) ↔ n > 3 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_equality_l3521_352145


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3521_352117

/-- The cost per kilogram of mangos -/
def mango_cost : ℝ := sorry

/-- The cost per kilogram of rice -/
def rice_cost : ℝ := sorry

/-- The cost per kilogram of flour -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3521_352117


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3521_352149

theorem imaginary_part_of_complex_division (i : ℂ) : 
  i * i = -1 → Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3521_352149


namespace NUMINAMATH_CALUDE_boat_speed_l3521_352135

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    the speed of the boat in still water is 8 km/h. -/
theorem boat_speed (b s : ℝ) 
    (h1 : b + s = 11)  -- Speed along the stream
    (h2 : b - s = 5)   -- Speed against the stream
    : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l3521_352135


namespace NUMINAMATH_CALUDE_dichromate_molecular_weight_l3521_352143

/-- The molecular weight of 9 moles of dichromate (Cr2O7^2-) -/
theorem dichromate_molecular_weight (Cr_weight O_weight : ℝ) 
  (h1 : Cr_weight = 52.00)
  (h2 : O_weight = 16.00) :
  9 * (2 * Cr_weight + 7 * O_weight) = 1944.00 := by
  sorry

end NUMINAMATH_CALUDE_dichromate_molecular_weight_l3521_352143


namespace NUMINAMATH_CALUDE_ratio_transformation_l3521_352108

theorem ratio_transformation (x : ℚ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l3521_352108


namespace NUMINAMATH_CALUDE_hyperbola_a_range_l3521_352169

theorem hyperbola_a_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) →
  (∃ m : ℝ, m = Real.tan (60 * π / 180) ∧ 
    ∀ x y : ℝ, y = m * (x - 4) → (x^2 / a^2 - y^2 / b^2 = 1 → x > 0) →
    ∃! p : ℝ × ℝ, p.1^2 / a^2 - p.2^2 / b^2 = 1 ∧ p.2 = m * (p.1 - 4) ∧ p.1 > 0) →
  0 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_a_range_l3521_352169


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l3521_352129

theorem roots_sum_reciprocal (α β : ℝ) : 
  α^2 - 10*α + 20 = 0 → β^2 - 10*β + 20 = 0 → 1/α + 1/β = 1/2 := by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l3521_352129


namespace NUMINAMATH_CALUDE_babysitting_theorem_l3521_352171

def babysitting_earnings (initial_charge : ℝ) (hours : ℕ) : ℝ :=
  let rec calc_earnings (h : ℕ) (prev_charge : ℝ) (total : ℝ) : ℝ :=
    if h = 0 then
      total
    else
      calc_earnings (h - 1) (prev_charge * 1.5) (total + prev_charge)
  calc_earnings hours initial_charge 0

theorem babysitting_theorem :
  babysitting_earnings 4 4 = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_theorem_l3521_352171


namespace NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l3521_352100

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The right angle measures 90 degrees -/
  is_right : right_angle = 90

/-- The number of right angles in a right triangle -/
def num_right_angles (t : RightTriangle) : ℕ := 1

theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  num_right_angles t = 1 := by
  sorry

#check right_triangle_has_one_right_angle

end NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l3521_352100


namespace NUMINAMATH_CALUDE_halfway_point_fractions_l3521_352172

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/8) (hb : b = 1/3) :
  (a + b) / 2 = 11/48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_fractions_l3521_352172


namespace NUMINAMATH_CALUDE_solve_equation_l3521_352148

theorem solve_equation (x y : ℝ) (h1 : x^2 - x + 6 = y - 6) (h2 : x = -6) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3521_352148


namespace NUMINAMATH_CALUDE_first_project_breadth_l3521_352173

/-- Represents a digging project with depth, length, breadth, and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  duration : ℝ

/-- The volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project with unknown breadth -/
def project1 (b : ℝ) : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := b,
  duration := 12
}

/-- The second digging project -/
def project2 : DiggingProject := {
  depth := 75,
  length := 20,
  breadth := 50,
  duration := 12
}

/-- The theorem stating that the breadth of the first project is 30 meters -/
theorem first_project_breadth :
  ∃ b : ℝ, volume (project1 b) = volume project2 ∧ b = 30 := by
  sorry


end NUMINAMATH_CALUDE_first_project_breadth_l3521_352173


namespace NUMINAMATH_CALUDE_expression_factorization_l3521_352170

theorem expression_factorization (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3521_352170


namespace NUMINAMATH_CALUDE_circus_ticket_price_l3521_352178

/-- Proves that the price of an upper seat ticket is $20 given the conditions of the circus ticket sales. -/
theorem circus_ticket_price :
  let lower_seat_price : ℕ := 30
  let total_tickets : ℕ := 80
  let total_revenue : ℕ := 2100
  let lower_seats_sold : ℕ := 50
  let upper_seats_sold : ℕ := total_tickets - lower_seats_sold
  let upper_seat_price : ℕ := (total_revenue - lower_seat_price * lower_seats_sold) / upper_seats_sold
  upper_seat_price = 20 := by sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l3521_352178


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l3521_352107

def weekly_allowance : ℚ := 240 / 100

def arcade_spending : ℚ := (3 / 5) * weekly_allowance

def remaining_after_arcade : ℚ := weekly_allowance - arcade_spending

def toy_store_spending : ℚ := (1 / 3) * remaining_after_arcade

def candy_store_spending : ℚ := remaining_after_arcade - toy_store_spending

theorem johns_candy_store_spending :
  candy_store_spending = 64 / 100 := by sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l3521_352107


namespace NUMINAMATH_CALUDE_pencils_per_package_l3521_352192

theorem pencils_per_package (pens_per_package : ℕ) (total_pens : ℕ) (pencil_packages : ℕ) : 
  pens_per_package = 12 → 
  total_pens = 60 → 
  total_pens / pens_per_package = pencil_packages →
  total_pens / pencil_packages = 12 :=
by sorry

end NUMINAMATH_CALUDE_pencils_per_package_l3521_352192


namespace NUMINAMATH_CALUDE_binary_decimal_base7_conversion_l3521_352132

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_decimal_base7_conversion :
  let binary := [true, false, true, true, false, true]
  binary_to_decimal binary = 45 ∧
  decimal_to_base7 45 = [6, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_decimal_base7_conversion_l3521_352132


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3521_352128

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n)
  (h2 : perp m n) 
  (h3 : perp_plane n α) 
  (h4 : ¬ subset m α) : 
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3521_352128


namespace NUMINAMATH_CALUDE_tom_customers_per_hour_l3521_352185

/-- The number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- The number of hours Tom worked -/
def hours_worked : ℝ := 8

/-- The bonus point percentage (20% = 0.2) -/
def bonus_percentage : ℝ := 0.2

/-- The total bonus points Tom earned -/
def total_bonus_points : ℝ := 16

theorem tom_customers_per_hour :
  customers_per_hour * hours_worked * bonus_percentage = total_bonus_points :=
by sorry

end NUMINAMATH_CALUDE_tom_customers_per_hour_l3521_352185


namespace NUMINAMATH_CALUDE_car_distance_proof_l3521_352142

/-- Proves that the distance covered by a car is 144 km given specific conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 6 →
  speed = 16 →
  time_factor = 3/2 →
  initial_time * time_factor * speed = 144 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l3521_352142


namespace NUMINAMATH_CALUDE_sum_a7_a9_eq_zero_l3521_352151

theorem sum_a7_a9_eq_zero (a : ℕ+ → ℤ) 
  (h : ∀ n : ℕ+, a n = 3 * n.val - 24) : 
  a 7 + a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_a7_a9_eq_zero_l3521_352151


namespace NUMINAMATH_CALUDE_hotel_to_ticket_ratio_l3521_352157

/-- Represents the trip expenses and calculates the ratio of hotel cost to ticket cost. -/
def tripExpenses (initialAmount ticketCost amountLeft : ℚ) : ℚ × ℚ := by
  -- Define total spent
  let totalSpent := initialAmount - amountLeft
  -- Define hotel cost
  let hotelCost := totalSpent - ticketCost
  -- Calculate the ratio
  let ratio := hotelCost / ticketCost
  -- Return the simplified ratio
  exact (1, 2)

/-- Theorem stating that the ratio of hotel cost to ticket cost is 1:2 for the given values. -/
theorem hotel_to_ticket_ratio :
  tripExpenses 760 300 310 = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_hotel_to_ticket_ratio_l3521_352157


namespace NUMINAMATH_CALUDE_composite_blackboard_theorem_l3521_352177

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ 1 < d ∧ d < n

def blackboard_numbers (n : ℕ) : Set ℕ :=
  {x | ∃ d, proper_divisor d n ∧ x = d + 1}

theorem composite_blackboard_theorem (n : ℕ) :
  is_composite n →
  (∃ m, blackboard_numbers n = {x | proper_divisor x m}) ↔
  n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_composite_blackboard_theorem_l3521_352177
