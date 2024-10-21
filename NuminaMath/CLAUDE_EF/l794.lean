import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_l794_79472

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem function_properties_and_triangle_side (B C : ℝ) :
  let ABC := { A : ℝ // 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi }
  Real.cos B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (b : ℝ), b = 8 / 3 ∧
    b / Real.sin C = Real.sqrt 6 / Real.sin B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_l794_79472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_bound_l794_79485

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n+1 => sequence_a n - n + 3

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i => sequence_a i)

def b (n : ℕ) : ℚ := (n + 1) / (S n - n + 2)

def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => b i)

theorem T_bound (n : ℕ) : T n < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_bound_l794_79485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l794_79494

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Area function for a triangle given two sides and the included angle -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.C

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) : 
  t.A = π/4 ∧ t.b = (Real.sqrt 2/2) * t.a →
  (t.B = π/6 ∧ 
   (t.a = Real.sqrt 2 → area t = (Real.sqrt 3 + 1)/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l794_79494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l794_79427

-- Define the plane
variable (Plane : Type)

-- Define points and sets
variable (A B T P C D K : Plane)
variable (Ω : Set Plane)  -- semicircle
variable (ω : Set Plane)  -- circle

-- Define properties and relations
variable (is_semicircle : Set Plane → Prop)
variable (is_circle : Set Plane → Prop)
variable (on_circle : Plane → Set Plane → Prop)
variable (on_arc : Plane → Set Plane → Prop)
variable (on_segment : Plane → Plane → Plane → Prop)
variable (different_sides : Plane → Plane → Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Plane → Plane → Prop)
variable (is_circumcenter : Plane → Plane → Plane → Plane → Prop)
variable (on_circumcircle : Plane → Plane → Plane → Plane → Prop)
variable (is_fixed_point : Plane → Prop)

-- State the theorem
theorem geometry_problem 
  (h1 : is_semicircle Ω)
  (h2 : on_arc T Ω)
  (h3 : is_circle ω)
  (h4 : on_circle A ω ∧ on_circle T ω)
  (h5 : on_arc P Ω)
  (h6 : on_circle C ω ∧ on_circle D ω)
  (h7 : on_segment C A P)
  (h8 : different_sides C D A B)
  (h9 : perpendicular C D A B)
  (h10 : is_circumcenter K C D P)
  : on_circumcircle K T D P ∧ is_fixed_point K :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_problem_l794_79427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l794_79414

theorem log_inequality (x : ℝ) : 
  let a := (2 : ℝ) ^ x
  let b := (4 : ℝ) ^ (2/3)
  (Real.log b / Real.log a ≤ 1) ↔ (x < 0 ∨ x ≥ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l794_79414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_cheating_theorem_l794_79498

/-- Represents the shop owner's pricing and cheating strategy -/
structure ShopOwnerStrategy where
  selling_cheat_percent : ℝ
  profit_margin_percent : ℝ
  buying_cheat_percent : ℝ

/-- Calculates the actual selling price based on the claimed cost price and profit margin -/
noncomputable def actual_selling_price (claimed_cost_price : ℝ) (profit_margin_percent : ℝ) : ℝ :=
  claimed_cost_price * (1 + profit_margin_percent / 100)

/-- Calculates the actual weight received when buying, accounting for cheating -/
noncomputable def actual_buying_weight (claimed_weight : ℝ) (buying_cheat_percent : ℝ) : ℝ :=
  claimed_weight * (1 - buying_cheat_percent / 100)

/-- Calculates the actual weight given when selling, accounting for cheating -/
noncomputable def actual_selling_weight (claimed_weight : ℝ) (selling_cheat_percent : ℝ) : ℝ :=
  claimed_weight * (1 - selling_cheat_percent / 100)

/-- Theorem: If the shop owner cheats by 20% while selling, has a 50% profit margin,
    and uses false weights for buying and selling, then the percentage by which
    he cheats while buying is also 20%. -/
theorem shop_owner_cheating_theorem (s : ShopOwnerStrategy)
  (h1 : s.selling_cheat_percent = 20)
  (h2 : s.profit_margin_percent = 50)
  : s.buying_cheat_percent = 20 := by
  sorry

#check shop_owner_cheating_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_cheating_theorem_l794_79498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_corn_peas_ratio_l794_79409

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a portion of land -/
structure LandPortion where
  total : ℝ
  corn : ℝ
  peas : ℝ

/-- The given problem setup -/
structure LandProblem where
  total_land : ℝ
  angela_barry_ratio : Ratio
  total_corn_peas_ratio : Ratio
  angela_corn_peas_ratio : Ratio
  angela_portion : LandPortion
  barry_portion : LandPortion

/-- The main theorem to prove -/
theorem barry_corn_peas_ratio 
  (prob : LandProblem) 
  (h1 : prob.total_land > 0)
  (h2 : prob.angela_barry_ratio = Ratio.mk 3 2)
  (h3 : prob.total_corn_peas_ratio = Ratio.mk 7 3)
  (h4 : prob.angela_corn_peas_ratio = Ratio.mk 4 1)
  (h5 : prob.angela_portion.total + prob.barry_portion.total = prob.total_land)
  (h6 : prob.angela_portion.corn + prob.angela_portion.peas = prob.angela_portion.total)
  (h7 : prob.barry_portion.corn + prob.barry_portion.peas = prob.barry_portion.total)
  (h8 : prob.angela_portion.corn + prob.barry_portion.corn = 
        prob.total_land * (prob.total_corn_peas_ratio.numerator / 
        (prob.total_corn_peas_ratio.numerator + prob.total_corn_peas_ratio.denominator)))
  (h9 : prob.angela_portion.peas + prob.barry_portion.peas = 
        prob.total_land * (prob.total_corn_peas_ratio.denominator / 
        (prob.total_corn_peas_ratio.numerator + prob.total_corn_peas_ratio.denominator)))
  : ∃ (r : Ratio), r = Ratio.mk 11 9 ∧ 
    r.numerator * prob.barry_portion.peas = r.denominator * prob.barry_portion.corn := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barry_corn_peas_ratio_l794_79409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_lens_cost_l794_79418

/-- The cost of a pair of soft contact lenses -/
def S : ℝ := sorry

/-- The cost of a pair of hard contact lenses -/
def H : ℝ := 85

/-- The number of pairs of hard contact lenses sold -/
def hard_pairs : ℕ := sorry

/-- The number of pairs of soft contact lenses sold -/
def soft_pairs : ℕ := hard_pairs + 5

/-- The total number of pairs of contact lenses sold -/
def total_pairs : ℕ := 11

/-- The total sales for pairs of contact lenses -/
def total_sales : ℝ := 1455

theorem soft_lens_cost :
  hard_pairs + soft_pairs = total_pairs ∧
  S * (soft_pairs : ℝ) + H * (hard_pairs : ℝ) = total_sales →
  S = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soft_lens_cost_l794_79418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l794_79410

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 1/2) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_from_sin_sum_diff_l794_79410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_minimum_l794_79452

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp (-x * Real.log 2)

theorem f_is_even_and_has_minimum : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_has_minimum_l794_79452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l794_79412

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  12321 ∣ x → 
  Int.gcd ((3*x+4)*(5*x+1)*(11*x+6)*(x+11)) x.natAbs = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_polynomial_and_multiple_l794_79412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ravis_income_difference_l794_79456

noncomputable def ravis_average : ℝ := 1025.68

noncomputable def greatest_even_integer_le (x : ℝ) : ℤ :=
  2 * ⌊x / 2⌋

theorem ravis_income_difference :
  ravis_average - (greatest_even_integer_le ravis_average : ℝ) = 1.68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ravis_income_difference_l794_79456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_is_40_point_5_l794_79440

/-- Tom's age in years -/
noncomputable def tom_age : ℝ := 40.5

/-- Antonette's age in years -/
noncomputable def antonette_age : ℝ := tom_age / 3

/-- The statement that Tom's age is 40.5 years, given the conditions -/
theorem tom_age_is_40_point_5 :
  (tom_age = 3 * antonette_age) ∧ (tom_age + antonette_age = 54) → tom_age = 40.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_is_40_point_5_l794_79440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l794_79478

theorem power_equality (x : ℝ) (h : (3 : ℝ)^(2*x) = 10) : (27 : ℝ)^(x+1) = 9 * Real.sqrt 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l794_79478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_shaded_region_volume_l794_79461

/-- Represents the shaded region in the problem -/
structure ShadeRegion where
  unit_squares : ℕ
  along_x_axis : Bool
  along_y_axis : Bool

/-- Calculates the volume of a cylinder -/
noncomputable def cylinder_volume (radius : ℝ) (height : ℝ) : ℝ := Real.pi * radius^2 * height

/-- Theorem: The volume of the solid formed by rotating the shaded region about the x-axis is 37π cubic units -/
theorem rotate_shaded_region_volume 
  (region : ShadeRegion) 
  (h1 : region.unit_squares = 11) 
  (h2 : region.along_x_axis = true) 
  (h3 : region.along_y_axis = true) :
  ∃ (v : ℝ), v = 37 * Real.pi ∧ v = cylinder_volume 5 1 + cylinder_volume 2 3 := by
  sorry

#check rotate_shaded_region_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_shaded_region_volume_l794_79461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_business_gain_percent_l794_79439

noncomputable section

def purchase_prices : List ℚ := [900, 1100, 1200, 800, 1000]
def repair_costs : List ℚ := [300, 400, 500, 350, 450]
def selling_prices : List ℚ := [1320, 1620, 1880, 1150, 1500]

def total_cost : ℚ := (List.zip purchase_prices repair_costs).map (fun (p, r) => p + r) |>.sum
def total_revenue : ℚ := selling_prices.sum
def total_gain : ℚ := total_revenue - total_cost
def gain_percent : ℚ := (total_gain / total_cost) * 100

theorem scooter_business_gain_percent :
  abs (gain_percent - 671/100) < 1/100 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_business_gain_percent_l794_79439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bahs_equal_to_yahs_l794_79416

def bah : ℕ → ℚ := sorry
def rah : ℕ → ℚ := sorry
def yah : ℕ → ℚ := sorry

axiom bah_to_rah : bah 20 = rah 40
axiom rah_to_yah : rah 12 = yah 24

theorem bahs_equal_to_yahs : bah 300 = yah 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bahs_equal_to_yahs_l794_79416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l794_79423

noncomputable def complex_number : ℂ := 1 / ((1 + Complex.I)^2 + 1)

theorem complex_number_in_fourth_quadrant :
  (complex_number.re > 0) ∧ (complex_number.im < 0) := by
  -- Expand the definition of complex_number
  have h : complex_number = (1/5 : ℝ) - (2/5 : ℝ) * Complex.I := by
    sorry
  -- Split into real and imaginary parts
  have real_part : complex_number.re = 1/5 := by
    sorry
  have imag_part : complex_number.im = -2/5 := by
    sorry
  -- Prove the conditions for the fourth quadrant
  constructor
  · -- Prove real part is positive
    rw [real_part]
    norm_num
  · -- Prove imaginary part is negative
    rw [imag_part]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_fourth_quadrant_l794_79423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctg_cx_equivalent_to_cx_l794_79446

theorem arctg_cx_equivalent_to_cx (c : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → |Real.arctan (c*x) / (c*x) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctg_cx_equivalent_to_cx_l794_79446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l794_79402

noncomputable def f (x : ℝ) := x + Real.log x / Real.log 10 - 3

theorem solution_interval :
  ∃ (a b : ℝ), a ∈ Set.Ioo 2 3 ∧ b ∈ Set.Ioo 2 3 ∧
  (∀ x, x ∈ Set.Ioo 2 3 → (f x = 0 ↔ x = a ∨ x = b)) ∧
  (∀ x, x < 2 → f x < 0) ∧
  (∀ x, x > 3 → f x > 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l794_79402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_far_from_visible_l794_79422

/-- A lattice point is visible if its coordinates are coprime -/
def visible (x y : ℤ) : Prop := Int.gcd x y = 1

/-- The distance between two points in ℤ × ℤ -/
noncomputable def distance (a b x y : ℤ) : ℝ :=
  Real.sqrt (((a - x : ℝ) ^ 2) + ((b - y : ℝ) ^ 2))

/-- There exists a lattice point at least 1995 units away from all visible points -/
theorem exists_point_far_from_visible : ∃ (a b : ℤ),
  ∀ (x y : ℤ), visible x y → distance a b x y ≥ 1995 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_far_from_visible_l794_79422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_of_N_values_l794_79428

theorem temperature_difference (N : ℤ) : 
  (∃ L : ℤ, 
    let M := L + N;
    let M_8pm := M - 10;
    let L_8pm := L + 6;
    abs (M_8pm - L_8pm) = 3) →
  (N = 13 ∨ N = 19) :=
by sorry

theorem product_of_N_values : 
  (∀ N : ℤ, (∃ L : ℤ, 
    let M := L + N;
    let M_8pm := M - 10;
    let L_8pm := L + 6;
    abs (M_8pm - L_8pm) = 3) →
  (N = 13 ∨ N = 19)) →
  13 * 19 = 247 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_difference_product_of_N_values_l794_79428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_with_ordered_angles_is_regular_l794_79431

-- Define a structure for a pentagon
structure Pentagon where
  A : Real
  B : Real
  C : Real
  D : Real
  E : Real

-- Define properties of the pentagon
def isConvex (p : Pentagon) : Prop := sorry
def isEquilateral (p : Pentagon) : Prop := sorry
def isRegular (p : Pentagon) : Prop := sorry

-- Define angle measurement
noncomputable def angle (p q r : Real) : Real := sorry

-- Theorem statement
theorem equilateral_pentagon_with_ordered_angles_is_regular 
  (p : Pentagon) 
  (h_convex : isConvex p) 
  (h_equilateral : isEquilateral p) 
  (h_angles : angle p.A p.B p.C ≥ angle p.B p.C p.D ∧ 
              angle p.B p.C p.D ≥ angle p.C p.D p.E ∧ 
              angle p.C p.D p.E ≥ angle p.D p.E p.A ∧ 
              angle p.D p.E p.A ≥ angle p.E p.A p.B) : 
  isRegular p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_pentagon_with_ordered_angles_is_regular_l794_79431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cream_ratio_l794_79462

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

def initial_cup1 : CupContents := { coffee := 6, cream := 0 }
def initial_cup2 : CupContents := { coffee := 2, cream := 4 }

def pour_half (from_cup : CupContents) (to_cup : CupContents) : (CupContents × CupContents) :=
  let total_from := from_cup.coffee + from_cup.cream
  let amount_poured := total_from / 2
  let new_from := { coffee := from_cup.coffee - (from_cup.coffee / total_from) * amount_poured,
                    cream := from_cup.cream - (from_cup.cream / total_from) * amount_poured }
  let new_to := { coffee := to_cup.coffee + (from_cup.coffee / total_from) * amount_poured,
                  cream := to_cup.cream + (from_cup.cream / total_from) * amount_poured }
  (new_from, new_to)

theorem coffee_cream_ratio : 
  let (cup1, cup2) := pour_half initial_cup1 initial_cup2
  let (final_cup1, _) := pour_half cup2 cup1
  final_cup1.cream / (final_cup1.coffee + final_cup1.cream) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_cream_ratio_l794_79462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_bounds_and_sqrt2_inequality_l794_79466

theorem root_bounds_and_sqrt2_inequality :
  let f : ℝ → ℝ := λ x ↦ x^2 - 198*x + 1
  let lower_bound : ℝ := 1/198
  let upper_bound : ℝ := 197.9949494949
  let repeated_decimal : ℝ := 1.41421356
  (∃ r : ℝ, f r = 0 ∧ lower_bound < r ∧ r < upper_bound) ∧
  (repeated_decimal < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_bounds_and_sqrt2_inequality_l794_79466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l794_79484

theorem f_derivative_at_zero (f : ℝ → ℝ) :
  (∀ x, f x = x^2 + 2*x*((deriv f) (-1))) →
  (deriv f) 0 = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l794_79484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l794_79473

/-- Represents the travel time and distance for a journey between two cities. -/
structure Journey where
  time : ℝ
  distance : ℝ

/-- Calculates the average speed given a journey. -/
noncomputable def averageSpeed (j : Journey) : ℝ := j.distance / j.time

/-- Theorem stating that Eddy's travel time is 3 hours given the conditions. -/
theorem eddy_travel_time 
  (eddy_journey : Journey)
  (freddy_journey : Journey)
  (h1 : freddy_journey.time = 4)
  (h2 : eddy_journey.distance = 480)
  (h3 : freddy_journey.distance = 300)
  (h4 : averageSpeed eddy_journey / averageSpeed freddy_journey = 2.1333333333333333) :
  eddy_journey.time = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_travel_time_l794_79473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_cleaning_time_l794_79411

/-- Represents the cleaning time for each person in a room -/
structure CleaningTime where
  lilly : ℚ
  fiona : ℚ
  jack : ℚ
  emily : ℚ

/-- The total cleaning time for all rooms in hours -/
def total_cleaning_time : ℚ := 12

/-- The number of rooms to be cleaned -/
def number_of_rooms : ℕ := 3

/-- Cleaning time distribution for room 1 -/
def room1 : CleaningTime :=
  { lilly := 1/4 - 1/12,  -- Lilly and Fiona combined is 1/4, assuming equal split
    fiona := 1/4 - 1/12,
    jack := 1/3,
    emily := 5/12 }  -- The rest

/-- Cleaning time distribution for room 2 -/
def room2 : CleaningTime :=
  { lilly := 1/4,  -- Lilly and Fiona split 50%, assuming equal split
    fiona := 1/4,
    jack := 1/4,
    emily := 1/4 }

/-- Cleaning time distribution for room 3 -/
def room3 : CleaningTime :=
  { lilly := 1/5,
    fiona := 1/5,
    jack := 1/5,
    emily := 2/5 }

/-- Emily's cleaning time in room 2 in minutes -/
def emily_room2_minutes : ℚ :=
  (total_cleaning_time / number_of_rooms) * room2.emily * 60

theorem emily_cleaning_time :
  emily_room2_minutes = 60 := by
  -- Expand the definition of emily_room2_minutes
  unfold emily_room2_minutes
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval emily_room2_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_cleaning_time_l794_79411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diameter_equation_l794_79421

/-- Given a circle and a chord, proves the equation of the line containing the diameter perpendicular to the chord -/
theorem perpendicular_diameter_equation 
  (m : ℝ) 
  (h_m : m < 3) 
  (A B : ℝ × ℝ) 
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + m = 0 → (x, y) ∈ Set.univ) 
  (h_midpoint : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 1) :
  ∃ (a b c : ℝ), a*x + b*y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -1 := by
  sorry

#check perpendicular_diameter_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_diameter_equation_l794_79421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l794_79445

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (scalar * w.1, scalar * w.2)

noncomputable def reflect (v w : ℝ × ℝ) : ℝ × ℝ :=
  let p := projection v w
  (2 * p.1 - v.1, 2 * p.2 - v.2)

theorem reflection_over_vector :
  let v : ℝ × ℝ := (3, -2)
  let w : ℝ × ℝ := (2, -1)
  reflect v w = (17/5, -6/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_over_vector_l794_79445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_average_speed_l794_79438

/-- Represents a half of Alice's journey -/
structure JourneyHalf where
  distance : ℚ
  time : ℚ

/-- Represents Alice's entire journey -/
structure Journey where
  first_half : JourneyHalf
  second_half : JourneyHalf

/-- Calculates the average speed of a journey -/
def average_speed (j : Journey) : ℚ :=
  (j.first_half.distance + j.second_half.distance) / (j.first_half.time + j.second_half.time)

/-- Alice's actual journey -/
def alices_journey : Journey :=
  { first_half := { distance := 24, time := 3 },
    second_half := { distance := 36, time := 3 } }

theorem alices_average_speed :
  average_speed alices_journey = 10 := by
  -- Unfold the definitions
  unfold average_speed alices_journey
  -- Simplify the arithmetic
  simp [add_div]
  -- Normalize to get the final result
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alices_average_speed_l794_79438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_form_rectangle_l794_79454

-- Define the two ellipses
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a line
def line (m x y : ℝ) : Prop := y = m * x

-- Define conjugate diameters
def conjugate_diameters (a b m m' : ℝ) : Prop := m' = -b^4 / (a^4 * m)

-- Define the slope of the tangent line
noncomputable def tangent_slope (a b m : ℝ) : ℝ := -b^2 / (a^2 * m)

theorem tangents_form_rectangle 
  (a b : ℝ) 
  (hpos : a > 0 ∧ b > 0) 
  (m m' : ℝ)
  (hm : m ≠ 0)
  (hm' : m' ≠ 0)
  (hconj : conjugate_diameters a b m m') :
  tangent_slope a b m * tangent_slope a b m' = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_form_rectangle_l794_79454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ssr_equals_0_03_l794_79455

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 1

-- Define the experimental data points
def data_points : List (ℝ × ℝ) := [(2, 4.9), (3, 7.1), (4, 9.1)]

-- Define the residual for a single data point
def calc_residual (point : ℝ × ℝ) : ℝ :=
  point.2 - regression_equation point.1

-- Define the sum of squared residuals
def sum_squared_residuals (points : List (ℝ × ℝ)) : ℝ :=
  (points.map (fun p => (calc_residual p) ^ 2)).sum

-- Theorem statement
theorem ssr_equals_0_03 :
  sum_squared_residuals data_points = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ssr_equals_0_03_l794_79455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l794_79471

/-- A circle passing through two points with its center on a given line -/
structure CircleWithCenter where
  center : ℝ × ℝ
  radius : ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  center_on_line : center.1 = center.2

/-- A line with a given slope -/
structure LineWithSlope where
  slope : ℝ
  y_intercept : ℝ

/-- The main theorem about the circle and its tangent line -/
theorem circle_and_tangent_line 
  (c : CircleWithCenter)
  (h1 : c.point1 = (0, 3))
  (h2 : c.point2 = (3, 2))
  (l : LineWithSlope)
  (h3 : l.slope = 1) :
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 5 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ (b : ℝ), b = Real.sqrt 10 ∨ b = -Real.sqrt 10) ∧ 
  (∀ (x y : ℝ), y = x + l.y_intercept ↔ y = x + l.y_intercept) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_line_l794_79471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_chords_theorem_l794_79487

/-- A parabola with equation y² = 4x -/
structure Parabola where
  x : ℝ → ℝ
  y : ℝ → ℝ
  eq : ∀ t, (y t)^2 = 4 * (x t)

/-- A chord of a parabola -/
structure Chord (p : Parabola) where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- The focus of a parabola y² = 4x is at (1, 0) -/
def focus (p : Parabola) : ℝ × ℝ := (1, 0)

/-- Two chords are perpendicular -/
def perpendicular {p : Parabola} (c1 c2 : Chord p) : Prop :=
  let (x1, y1) := c1.start
  let (x2, y2) := c1.finish
  let (x3, y3) := c2.start
  let (x4, y4) := c2.finish
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

/-- The length of a chord -/
noncomputable def chord_length {p : Parabola} (c : Chord p) : ℝ :=
  let (x1, y1) := c.start
  let (x2, y2) := c.finish
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem to be proved -/
theorem parabola_perpendicular_chords_theorem (p : Parabola) 
  (c1 c2 : Chord p) 
  (h1 : c1.start = focus p ∨ c1.finish = focus p) 
  (h2 : c2.start = focus p ∨ c2.finish = focus p) 
  (h3 : perpendicular c1 c2) :
  1 / chord_length c1 + 1 / chord_length c2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_chords_theorem_l794_79487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vanya_can_win_l794_79458

/-- Represents a player in the game -/
inductive Player
| Vanya
| Seryozha

/-- Represents a chip color -/
inductive ChipColor
| Black
| White

/-- Represents a position on the 4x6 grid -/
structure Position :=
(row : Nat)
(col : Nat)

/-- Represents the game state -/
structure GameState :=
(vanyaChips : List Position)
(seryozhaChips : List Position)
(currentPlayer : Player)

/-- Defines a valid move in the game -/
def validMove (start finish : Position) : Prop :=
  start.col = finish.col ∧ finish.row = start.row - 1 ∧ start.row > 0 ∧ start.row ≤ 4 ∧ finish.row ≥ 1 ∧ finish.row ≤ 4

/-- Defines when a black chip is captured -/
def isCaptured (blackChip : Position) (whiteChips : List Position) : Prop :=
  ∃ (w1 w2 : Position), w1 ∈ whiteChips ∧ w2 ∈ whiteChips ∧
    ((w1.row = w2.row ∧ w1.row = blackChip.row ∧ 
      ((w1.col < blackChip.col ∧ blackChip.col < w2.col) ∨ 
       (w2.col < blackChip.col ∧ blackChip.col < w1.col))) ∨
     (w1.row = blackChip.row + 1 ∧ w2.row = blackChip.row - 1 ∧
      ((w1.col = blackChip.col - 1 ∧ w2.col = blackChip.col + 1) ∨
       (w1.col = blackChip.col + 1 ∧ w2.col = blackChip.col - 1))))

/-- Theorem: Vanya can always move both chips to the bottom row -/
theorem vanya_can_win (initialState : GameState) :
  initialState.vanyaChips.length = 2 ∧
  initialState.seryozhaChips.length = 2 ∧
  (∀ p ∈ initialState.vanyaChips, p.row = 4) ∧
  (∀ p ∈ initialState.seryozhaChips, p.row = 4) ∧
  initialState.currentPlayer = Player.Vanya →
  ∃ (finalState : GameState),
    (∀ p ∈ finalState.vanyaChips, p.row = 1) ∧
    finalState.vanyaChips.length = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vanya_can_win_l794_79458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encoded_xyz_value_l794_79475

/-- Represents the set of symbols used in the encoding --/
inductive EncSymbol
  | V | W | X | Y | Z | A | B

/-- Represents a base-7 number encoded with symbols --/
def EncodedNumber := List EncSymbol

/-- Converts a EncSymbol to its corresponding base-7 digit --/
def symbolToDigit : EncSymbol → Fin 7 := sorry

/-- Converts an EncodedNumber to its base-10 value --/
def toBase10 (n : EncodedNumber) : ℕ := sorry

/-- The main theorem to prove --/
theorem encoded_xyz_value :
  ∃ (vxy vyb vza : EncodedNumber),
    (vxy.length = 3 ∧ vyb.length = 3 ∧ vza.length = 3) ∧
    (vxy.head? = some EncSymbol.V ∧ vyb.head? = some EncSymbol.V ∧ vza.head? = some EncSymbol.V) ∧
    (toBase10 vyb = toBase10 vxy + 3) ∧
    (toBase10 vza = toBase10 vyb - 1) ∧
    toBase10 [EncSymbol.X, EncSymbol.Y, EncSymbol.Z] = 288 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_encoded_xyz_value_l794_79475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_l794_79469

/-- Given that C = (4, 7) is the midpoint of AB, where A = (2, 10) and B = (x, y), prove that x + y = 10 -/
theorem midpoint_sum (x y : ℝ) : 
  (4, 7) = ((2 + x) / 2, (10 + y) / 2) → x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_l794_79469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_student_percentage_theorem_l794_79479

/-- Calculates the percentage of local students across arts, science, and commerce departments. -/
def localStudentPercentage (artsTotal : ℕ) (artsLocalPercentage : ℚ)
                           (scienceTotal : ℕ) (scienceLocalPercentage : ℚ)
                           (commerceTotal : ℕ) (commerceLocalPercentage : ℚ) : ℚ :=
  let artsLocal := (artsLocalPercentage * artsTotal) / 100
  let scienceLocal := (scienceLocalPercentage * scienceTotal) / 100
  let commerceLocal := (commerceLocalPercentage * commerceTotal) / 100
  let totalLocal := artsLocal + scienceLocal + commerceLocal
  let totalStudents := (artsTotal + scienceTotal + commerceTotal : ℚ)
  (totalLocal / totalStudents) * 100

/-- The percentage of local students across arts, science, and commerce departments is approximately 52.74%. -/
theorem local_student_percentage_theorem :
  ∃ ε > 0, |localStudentPercentage 400 50 100 25 120 85 - 52.74| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_student_percentage_theorem_l794_79479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_equal_l794_79400

/-- Given a real number k satisfying 0 < k < 9, the focal distances of the hyperbolas
    represented by (x^2/25) - (y^2/(9-k)) = 1 and (x^2/(25-k)) - (y^2/9) = 1 are equal. -/
theorem hyperbola_focal_distances_equal (k : ℝ) (h1 : 0 < k) (h2 : k < 9) :
  Real.sqrt (34 - k) = Real.sqrt (34 - k) := by
  -- The focal distances are already equal by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_equal_l794_79400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l794_79465

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x - a / x

-- State the theorem
theorem function_properties (a : ℝ) :
  (f a 2 = 9/2) →
  (a = -1) ∧
  (∀ x : ℝ, x ≠ 0 → f (-1) (-x) = -(f (-1) x)) ∧
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f (-1) x₁ < f (-1) x₂) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l794_79465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_projection_l794_79451

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Definition of an acute scalene triangle -/
def isAcuteScalene (t : Triangle) : Prop := sorry

/-- Definition of a circumcircle of a triangle -/
def hasCircumcircle (t : Triangle) (c : Circle) : Prop := sorry

/-- Definition of tangent points on a circle -/
def areTangentPoints (c : Circle) (P Q : Point) : Prop := sorry

/-- Definition of intersection point of two lines -/
noncomputable def intersectionPoint (P1 Q1 P2 Q2 : Point) : Point := sorry

/-- Definition of projection of a point onto a line -/
noncomputable def projection (P : Point) (L1 L2 : Point) : Point := sorry

/-- Definition of distance between two points -/
noncomputable def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem triangle_tangent_projection (t : Triangle) (ω : Circle) (T X Y : Point) :
  isAcuteScalene t →
  hasCircumcircle t ω →
  areTangentPoints ω t.B t.C →
  T = intersectionPoint t.B (projection t.B t.A t.C) t.C (projection t.C t.A t.B) →
  X = projection T t.A t.B →
  Y = projection T t.A t.C →
  distance T t.B = 20 →
  distance T t.C = 20 →
  distance t.B t.C = 30 →
  (distance T X)^2 + (distance T Y)^2 + (distance X Y)^2 = 2020 →
  ∃ x y : ℝ,
    x = distance t.B X ∧
    y = distance t.C Y ∧
    (distance X Y)^2 = 1220 - x^2 - y^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_projection_l794_79451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totalAmount_correct_l794_79480

/-- The total amount accumulated after annual deposits with compound interest -/
noncomputable def totalAmount (a p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

/-- Theorem stating the correctness of the totalAmount function -/
theorem totalAmount_correct (a p : ℝ) (hp : p ≠ 0) :
  let S := totalAmount a p
  S = (a * (1 + p)) * (1 - (1 + p)^7) / (1 - (1 + p)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_totalAmount_correct_l794_79480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_specific_l794_79457

/-- An arithmetic progression with 100 terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
noncomputable def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

/-- Sum of squares of first n terms of an arithmetic progression -/
noncomputable def sum_squares (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) * ap.a^2 + ap.d^2 * ((n - 1 : ℝ) * (n : ℝ) * (2 * n - 1 : ℝ)) / 6 + 
  ap.a * ap.d * (n : ℝ) * ((n - 1 : ℝ)) / 3

/-- Theorem about the sum of squares in a specific arithmetic progression -/
theorem sum_squares_specific (ap : ArithmeticProgression) :
  sum_n ap 100 = 150 ∧ sum_n { a := ap.a + ap.d, d := 2 * ap.d } 50 = 50 →
  sum_squares ap 100 = 241950 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squares_specific_l794_79457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l794_79406

-- Define the hyperbola and its properties
noncomputable def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

-- Define the asymptote
noncomputable def Asymptote (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}

-- Define eccentricity
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Asymptote (Real.sqrt 3) ⊆ Hyperbola a b → Eccentricity a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l794_79406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_tree_square_l794_79407

theorem invisible_tree_square (n : ℕ+) :
  ∃ N M : ℤ, ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → Int.gcd (N + i) (M + j) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invisible_tree_square_l794_79407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dist_C3_to_l_l794_79499

open Real

/-- The curve C3 -/
def C3 (x y : ℝ) : Prop := y^2 / 3 + x^2 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := 2 * Real.sqrt 3 * x + 2 * y + 1 = 0

/-- The distance function from a point to the line l -/
noncomputable def dist_to_l (x y : ℝ) : ℝ :=
  abs (2 * Real.sqrt 3 * x + 2 * y + 1) / Real.sqrt (12 + 4)

/-- The theorem stating the maximum distance from C3 to l -/
theorem max_dist_C3_to_l :
  ∃ (max_dist : ℝ), max_dist = (1 + 2 * Real.sqrt 6) / 4 ∧
  ∀ (x y : ℝ), C3 x y → dist_to_l x y ≤ max_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dist_C3_to_l_l794_79499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l794_79433

/-- Given a line and a hyperbola, if a point P satisfies certain conditions,
    then the eccentricity of the hyperbola is √5/2 -/
theorem hyperbola_eccentricity
  (m a b : ℝ)
  (hm : m ≠ 0)
  (ha : a > 0)
  (hb : b > 0)
  (line : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → Prop)
  (asymptote_pos asymptote_neg : ℝ → ℝ)
  (A B : ℝ × ℝ)
  (hline : ∀ x y, line x y ↔ x - 3*y + m = 0)
  (hhyperbola : ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1)
  (hasymptote_pos : ∀ x, asymptote_pos x = (b/a) * x)
  (hasymptote_neg : ∀ x, asymptote_neg x = -(b/a) * x)
  (hA : line A.1 A.2 ∧ A.2 = asymptote_pos A.1)
  (hB : line B.1 B.2 ∧ B.2 = asymptote_neg B.1)
  (hP : ‖(A.1 - m, A.2)‖ = ‖(B.1 - m, B.2)‖) :
  ∃ c, c^2 = a^2 + b^2 ∧ (Real.sqrt 5) / 2 = c/a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l794_79433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_square_l794_79442

theorem product_divisible_by_square (S : Finset ℕ) : 
  S.card = 6 → (∀ n, n ∈ S → n ≥ 1 ∧ n ≤ 10) → (∀ a b, a ∈ S → b ∈ S → a ≠ b) →
  ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ S.prod id :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_square_l794_79442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_savings_proof_l794_79430

def shirt_savings_problem (total_cost saved_amount weekly_savings : ℚ) : ℕ :=
  -- Calculate the remaining amount to save
  let remaining := total_cost - saved_amount
  -- Calculate the number of weeks needed (as a rational number)
  let weeks_rational := remaining / weekly_savings
  -- Convert the rational number to the ceiling natural number
  (weeks_rational.ceil).toNat

-- Theorem statement
theorem shirt_savings_proof :
  shirt_savings_problem 3 1.5 0.5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_savings_proof_l794_79430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l794_79464

def p (x : ℝ) : Prop := |4 * x - 1| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

def necessary_not_sufficient (P Q : ℝ → Prop) : Prop :=
  (∀ x, Q x → P x) ∧ ∃ x, P x ∧ ¬(Q x)

theorem range_of_a (a : ℝ) :
  (necessary_not_sufficient (fun x => ¬(p x)) (fun x => ¬(q x a))) →
  -1/2 ≤ a ∧ a ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l794_79464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_five_l794_79424

-- Define the equations of line l₁, curve C, and line l₂
noncomputable def l₁ (x y : ℝ) : Prop := y = Real.sqrt 3 * x

noncomputable def C (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.sqrt 3 * Real.cos φ, Real.sqrt 3 * Real.sin φ)

noncomputable def l₂ (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))
noncomputable def B : ℝ × ℝ := (-3 * Real.cos (Real.pi / 3), -3 * Real.sin (Real.pi / 3))

-- State the theorem
theorem distance_AB_is_five :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_five_l794_79424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_ratio_bound_l794_79434

/-- A tetrahedron with two edges of length a and four edges of length b -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  a_positive : 0 < a
  b_positive : 0 < b

/-- The ratio of edge lengths in a tetrahedron -/
noncomputable def edge_ratio (t : Tetrahedron) : ℝ := t.a / t.b

/-- The upper bound for the edge ratio -/
noncomputable def upper_bound : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 2

theorem tetrahedron_edge_ratio_bound (t : Tetrahedron) : 
  0 < edge_ratio t ∧ edge_ratio t < upper_bound := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_ratio_bound_l794_79434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_two_four_six_l794_79432

/-- The set of numbers from 1 to 120 -/
def S : Finset ℕ := Finset.range 120

/-- A number is a multiple of 2, 4, or 6 -/
def is_multiple (n : ℕ) : Prop := n % 2 = 0 ∨ n % 4 = 0 ∨ n % 6 = 0

/-- The subset of S containing multiples of 2, 4, or 6 -/
def T : Finset ℕ := S.filter (fun n => n % 2 = 0 ∨ n % 4 = 0 ∨ n % 6 = 0)

theorem probability_multiple_two_four_six :
  (T.card : ℚ) / S.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_two_four_six_l794_79432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_two_thirds_l794_79459

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of a line
def line_slope (m : ℝ) (x y : ℝ) : Prop := y = m * x + (12 / 6)

-- Theorem: The slope of the line given by 4x - 6y = 12 is 2/3
theorem slope_is_two_thirds :
  ∃ (m : ℝ), m = 2/3 ∧ ∀ (x y : ℝ), line_equation x y → line_slope m x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_two_thirds_l794_79459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l794_79449

/-- Calculates the area of a quadrilateral given its vertices -/
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- Implementation of the Shoelace formula
  sorry

/-- The area of a quadrilateral with vertices (0,0), (2,4), (6,0), and (2,6) is 6 -/
theorem quadrilateral_area : 
  let vertices : List (ℝ × ℝ) := [(0, 0), (2, 4), (6, 0), (2, 6)]
  area_of_quadrilateral vertices = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l794_79449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_fifth_product_l794_79435

-- Define the geometric sequence
noncomputable def geometric_sequence : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to avoid the missing case error
  | 1 => Real.rpow 5 (1/2)
  | 2 => Real.rpow 5 (1/4)
  | 3 => Real.rpow 5 (1/7)
  | _ => 1  -- Default case for other natural numbers

-- State the theorem
theorem fourth_fifth_product :
  (geometric_sequence 4) * (geometric_sequence 5) = Real.rpow 5 (17/72) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_fifth_product_l794_79435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_96_l794_79405

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to a horizontal line -/
def distanceToHorizontalLine (p : Point) (lineY : ℝ) : ℝ :=
  |p.y - lineY|

theorem sum_of_coordinates_is_96 (points : Finset Point) : 
  points.card = 4 ∧ 
  (∀ p ∈ points, distanceToHorizontalLine p 15 = 7) ∧
  (∀ p ∈ points, distance p ⟨9, 15⟩ = 15) →
  (points.sum (λ p => p.x + p.y)) = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_96_l794_79405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_y_equals_one_l794_79488

/-- Slope angle of a line -/
def SlopeAngle (f : ℝ → ℝ) : ℝ := sorry

/-- A line parallel to the x-axis has a slope angle of 0° -/
axiom slope_angle_parallel_x_axis (y : ℝ) : SlopeAngle (λ x : ℝ => y) = 0

/-- The slope angle of the line y = 1 is 0° -/
theorem slope_angle_y_equals_one : SlopeAngle (λ x : ℝ => 1) = 0 := by
  exact slope_angle_parallel_x_axis 1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_y_equals_one_l794_79488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_highest_difference_l794_79492

/-- Represents the sales data for a single month --/
structure MonthSales where
  drummers : ℝ
  bugle : ℝ
  flute : ℝ

/-- Calculates the percentage difference between the highest sales and the average of the other two --/
noncomputable def percentageDifference (sales : MonthSales) : ℝ :=
  let max_sales := max (max sales.drummers sales.bugle) sales.flute
  let other_two_avg := (sales.drummers + sales.bugle + sales.flute - max_sales) / 2
  (max_sales - other_two_avg) / other_two_avg * 100

/-- Represents the sales data for all months --/
def allSales : List MonthSales := [
  ⟨8, 5, 6⟩,  -- February
  -- Add other months' data here
]

theorem february_highest_difference :
  ∀ month ∈ allSales,
    percentageDifference (MonthSales.mk 8 5 6) ≥ percentageDifference month := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_highest_difference_l794_79492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_on_nonpos_main_theorem_l794_79491

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem even_function_decreasing_on_nonpos (f : ℝ → ℝ) 
  (h_even : even_function f) (h_incr : increasing_on_nonneg f) :
  ∀ x y, x ≤ y → y ≤ 0 → f y ≤ f x :=
sorry

theorem main_theorem (f : ℝ → ℝ) (a : ℝ)
  (h_even : even_function f)
  (h_incr : increasing_on_nonneg f)
  (h_ineq : ∀ x, 1/2 ≤ x → x ≤ 1 → f (a*x + 1) ≤ f (x - 2)) :
  -2 ≤ a ∧ a ≤ 0 :=
sorry

#check main_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_on_nonpos_main_theorem_l794_79491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l794_79415

/-- Parabola type -/
structure Parabola where
  f : ℝ × ℝ → ℝ
  focus : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.f point = 0

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Minimum distance sum for parabola -/
theorem min_distance_sum_parabola :
  let p : Parabola := { f := fun (x : ℝ × ℝ) => x.2^2 - 4*x.1, focus := (1, 0) }
  let a : ℝ × ℝ := (2, 2)
  ∀ point : ℝ × ℝ, PointOnParabola p point →
    ∃ min_val : ℝ, min_val = 3 ∧
    ∀ other_point : ℝ × ℝ, PointOnParabola p other_point →
      distance point a + distance point p.focus ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l794_79415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l794_79468

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -Real.sqrt 3; Real.sqrt 3, 2]

theorem matrix_power_four :
  A ^ 4 = !![-(49/2), -(49 * Real.sqrt 3)/2; (49 * Real.sqrt 3)/2, -(49/2)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l794_79468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_q_range_p_or_q_false_range_l794_79496

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
-- We'll use a placeholder for IsHyperbola since it's not defined in the standard library
def IsHyperbola (x y : ℝ) : Prop := sorry

def q (m : ℝ) : Prop := ∀ x y : ℝ, x^2/(1-2*m) + y^2/(m+2) = 1 → IsHyperbola x y

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m ≤ -1 ∨ m ≥ 2 := by
  sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -2 ∨ m > 1/2 := by
  sorry

-- Theorem for the range of m when (p ∨ q) is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -1 < m ∧ m ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_q_range_p_or_q_false_range_l794_79496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_zero_a_range_for_three_zeros_l794_79408

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (x - 2*a)*(a - x) else Real.sqrt x + a - 1

-- Part 1
theorem range_when_a_zero :
  let S := {y | ∃ x ∈ Set.Icc 0 4, f 0 x = y}
  S = Set.Icc (-1) 1 := by sorry

-- Part 2
theorem a_range_for_three_zeros :
  {a : ℝ | ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)} =
  Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_a_zero_a_range_for_three_zeros_l794_79408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_film_radius_is_10_sqrt_10_l794_79426

/-- The radius of a circular film formed by pouring liquid from a cylinder onto water -/
noncomputable def circular_film_radius (cylinder_radius : ℝ) (cylinder_height : ℝ) (film_thickness : ℝ) : ℝ :=
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  Real.sqrt (cylinder_volume / (Real.pi * film_thickness))

/-- Theorem stating that the radius of the circular film is 10√10 cm -/
theorem circular_film_radius_is_10_sqrt_10 :
  circular_film_radius 5 8 0.2 = 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_film_radius_is_10_sqrt_10_l794_79426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_together_proof_l794_79420

noncomputable def work_together (time1 time2 : ℝ) : ℝ :=
  1 / (1 / time1 + 1 / time2)

theorem work_together_proof (time1 time2 : ℝ) (h1 : time1 = 24) (h2 : time2 = 40) :
  work_together time1 time2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_together_proof_l794_79420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_problem_l794_79453

/-- Calculates the volume of a cylinder not occupied by cones -/
noncomputable def unoccupied_volume (r h h_cone : ℝ) : ℝ :=
  let v_cylinder := Real.pi * r^2 * h
  let v_full_cone := (1/3) * Real.pi * r^2 * h_cone
  let v_half_cone := (1/2) * v_full_cone
  v_cylinder - (v_full_cone + v_half_cone)

/-- Theorem stating the unoccupied volume in the given problem -/
theorem unoccupied_volume_problem :
  unoccupied_volume 10 30 10 = 2500 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_problem_l794_79453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_proof_l794_79486

theorem function_identity_proof {T : Type} [Field T] [Inhabited T] :
  ∀ (g : T → T), 
    (∀ (x y : T), x + y ≠ 0 → g x + g y = 4 * g ((x * y) / (g (x + y)))) →
    ∀ (x : T), x ≠ 0 → g x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_proof_l794_79486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_characterization_l794_79404

/-- The range of a polynomial function from ℝ² to ℝ -/
inductive PolynomialRange : Set ℝ → Prop where
  | full : PolynomialRange Set.univ
  | ray_right : (a : ℝ) → PolynomialRange { x | a ≤ x }
  | ray_left : (a : ℝ) → PolynomialRange { x | x ≤ a }
  | open_ray_right : (a : ℝ) → PolynomialRange { x | a < x }
  | open_ray_left : (a : ℝ) → PolynomialRange { x | x < a }
  | singleton : (a : ℝ) → PolynomialRange {a}

/-- A predicate to check if a function is a polynomial -/
def IsPolynomial (f : ℝ × ℝ → ℝ) : Prop := sorry

/-- The range of any polynomial function from ℝ² to ℝ is one of the PolynomialRange cases -/
theorem polynomial_range_characterization (f : ℝ × ℝ → ℝ) (hf : IsPolynomial f) :
  ∃ (S : Set ℝ), PolynomialRange S ∧ Set.range f = S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_characterization_l794_79404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_is_32pi_l794_79463

noncomputable section

def small_circle_diameter : ℝ := 4

def small_circle_radius : ℝ := small_circle_diameter / 2

def large_circle_radius : ℝ := 3 * small_circle_radius

def small_circle_area : ℝ := Real.pi * small_circle_radius ^ 2

def large_circle_area : ℝ := Real.pi * large_circle_radius ^ 2

def gray_area : ℝ := large_circle_area - small_circle_area

theorem gray_area_is_32pi : gray_area = 32 * Real.pi := by
  -- Expand the definitions
  unfold gray_area large_circle_area small_circle_area large_circle_radius small_circle_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gray_area_is_32pi_l794_79463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2_floor_is_4_l794_79495

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x - 1)) + (10 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem F_2_floor_is_4 : ⌊F 2⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_2_floor_is_4_l794_79495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_iff_z_second_quadrant_iff_l794_79444

/-- The complex number z as a function of real x -/
def z (x : ℝ) : ℂ := Complex.mk (x^2 - 2*x - 3) (x^2 + 3*x + 2)

theorem z_real_iff (x : ℝ) : (z x).im = 0 ↔ x = -2 ∨ x = -1 := by sorry

theorem z_second_quadrant_iff (x : ℝ) : 
  (z x).re < 0 ∧ (z x).im > 0 ↔ -1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_real_iff_z_second_quadrant_iff_l794_79444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l794_79490

theorem point_on_circle (t : ℝ) : 
  (Real.cos t + 1 - 1)^2 + (Real.sin t + 1 - 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l794_79490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_log_range_l794_79477

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a / x) - 1) / Real.log a

-- State the theorem
theorem monotone_increasing_log_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (2/5 : ℝ), StrictMono (f a)) →
  a ∈ Set.Ioo (2/5 : ℝ) (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_log_range_l794_79477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l794_79425

def sequence_a : ℕ → ℝ
  | 0 => 2
  | (n + 1) => (121 * (sequence_a n)^3)^(1/3)

theorem a_50_value : sequence_a 49 = 2 * 11^49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l794_79425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_weakly_decreasing_a_range_k_range_l794_79448

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 + x)

def weakly_decreasing (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  (∀ x ∈ D, ∀ y ∈ D, x ≤ y → f x ≥ f y) ∧
  (∀ x ∈ D, ∀ y ∈ D, x ≤ y → x * f x ≤ y * f y)

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f x + k * |x| - 1

-- Theorem 1
theorem f_weakly_decreasing :
  weakly_decreasing f (Set.Ici 0) := by
  sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, a / x ≤ f x ∧ f x ≤ (a + 4) / (2 * x)) ↔
  a ∈ Set.Icc (-1) (1/2) := by
  sorry

-- Theorem 3
theorem k_range (k : ℝ) :
  (∃ x y, x ∈ Set.Icc 0 3 ∧ y ∈ Set.Icc 0 3 ∧ x ≠ y ∧ g k x = 0 ∧ g k y = 0) ↔
  k ∈ Set.Icc (1/6) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_weakly_decreasing_a_range_k_range_l794_79448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_area_l794_79401

-- Define the circle radius
def circle_radius : ℝ := 1

-- Define the number of shaded circles
def num_shaded_circles : ℕ := 4

-- Define the number of spaces between circles
def num_spaces : ℕ := 3

-- Define the area of a single circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the area of a single space between circles
noncomputable def space_area : ℝ := 4 - Real.pi

-- Theorem statement
theorem shaded_region_area :
  (num_shaded_circles : ℝ) * circle_area circle_radius + 
  (num_spaces : ℝ) * space_area = 12 + Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_area_l794_79401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_value_l794_79497

theorem max_cos_x_value (x y z : Real) 
  (h1 : Real.sin x = Real.cos y)
  (h2 : Real.sin y = 1 / Real.cos z)
  (h3 : Real.sin z = Real.cos x) :
  ∃ (max_cos_x : Real), max_cos_x = 1 ∧ ∀ x', Real.cos x' ≤ max_cos_x :=
by
  -- We'll use 1 as the maximum value of cosine
  use 1
  constructor
  -- Prove that 1 is a possible value for cos x
  · sorry
  -- Prove that 1 is the maximum value for cosine
  · intro x'
    exact Real.cos_le_one x'


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_x_value_l794_79497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l794_79413

theorem terminal_side_quadrant (α : Real) : 
  (Real.sin α * Real.cos α > 0) → 
  (((0 < α ∧ α < Real.pi / 2) ∨ (Real.pi < α ∧ α < 3 * Real.pi / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_quadrant_l794_79413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l794_79482

def complex_bisector (z : ℂ) : Prop :=
  (z.re + z.im : ℝ) = 0

theorem complex_problem (a b : ℝ) (h_a_pos : a > 0) :
  let z : ℂ := a + b * Complex.I
  (Complex.abs z = Real.sqrt 10) →
  complex_bisector ((1 - 2 * Complex.I) * z) →
  (z = 3 + Complex.I ∧ 
   ∀ m : ℝ, (Complex.I * (Complex.ofReal 2 + (Complex.ofReal m + Complex.I) / (1 - Complex.I))).re = 0 → m = -5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l794_79482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l794_79493

/-- Calculates simple interest -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem statement -/
theorem interest_calculation (principal rate time : ℝ) : 
  principal = 9000 ∧ 
  time = 2 ∧ 
  simple_interest principal rate time = 900 →
  compound_interest principal rate time = 922.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l794_79493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_total_interest_l794_79489

/-- Calculates the total interest earned from two investments with different interest rates -/
theorem calculate_total_interest 
  (total_investment : ℝ) 
  (investment_1 : ℝ) 
  (interest_rate_1 : ℝ) 
  (interest_rate_2 : ℝ) 
  (h1 : total_investment = 9000)
  (h2 : investment_1 = 4000)
  (h3 : interest_rate_1 = 0.08)
  (h4 : interest_rate_2 = 0.09) :
  investment_1 * interest_rate_1 + (total_investment - investment_1) * interest_rate_2 = 770 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_total_interest_l794_79489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_addition_formula_cos_triple_angle_cos_quadruple_angle_l794_79450

/-- Cosine addition formula for any real number n and angle α -/
theorem cos_addition_formula (n : ℝ) (α : ℝ) :
  Real.cos ((n + 1) * α) = 2 * Real.cos α * Real.cos (n * α) - Real.cos ((n - 1) * α) := by sorry

/-- Cosine of triple angle formula -/
theorem cos_triple_angle (α : ℝ) :
  Real.cos (3 * α) = 4 * (Real.cos α)^3 - 3 * Real.cos α := by sorry

/-- Cosine of quadruple angle formula -/
theorem cos_quadruple_angle (α : ℝ) :
  Real.cos (4 * α) = 8 * (Real.cos α)^4 - 8 * (Real.cos α)^2 + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_addition_formula_cos_triple_angle_cos_quadruple_angle_l794_79450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_charitable_amount_l794_79467

/-- Calculates the amount Jill uses for gifts and charitable causes based on her salary and expense allocations -/
theorem jill_charitable_amount (net_salary : ℝ) (discretionary_ratio vacation_ratio savings_ratio social_ratio : ℝ) : 
  net_salary = 3700 →
  discretionary_ratio = 1/5 →
  vacation_ratio = 0.30 →
  savings_ratio = 0.20 →
  social_ratio = 0.35 →
  (net_salary * discretionary_ratio - 
   (net_salary * discretionary_ratio * vacation_ratio + 
    net_salary * discretionary_ratio * savings_ratio + 
    net_salary * discretionary_ratio * social_ratio)) = 111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_charitable_amount_l794_79467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ages_is_seventeen_l794_79443

/-- Represents the age statistics and criteria for job applicants -/
structure AgeStatistics where
  average_age : ℕ
  standard_deviation : ℕ
  acceptable_range : Set ℕ

/-- Calculates the maximum number of different integer ages within the acceptable range -/
def max_different_ages (stats : AgeStatistics) : ℕ :=
  Finset.card (Finset.range (stats.average_age + stats.standard_deviation + 1) ∩
               Finset.filter (λ x => x ≥ stats.average_age - stats.standard_deviation) (Finset.range (stats.average_age + stats.standard_deviation + 1)))

/-- The specific age statistics for the given problem -/
def job_applicant_stats : AgeStatistics :=
  { average_age := 31
    standard_deviation := 8
    acceptable_range := Set.Icc (31 - 8) (31 + 8) }

/-- Theorem stating that the maximum number of different ages is 17 -/
theorem max_ages_is_seventeen :
  max_different_ages job_applicant_stats = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ages_is_seventeen_l794_79443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_weekly_consumption_l794_79429

noncomputable def thermos_size : ℝ := 20

noncomputable def original_consumption (day : Nat) : ℝ :=
  match day with
  | 1 | 3 => 2 * thermos_size  -- Monday and Wednesday
  | 2 | 4 => 3 * thermos_size  -- Tuesday and Thursday
  | 5 => thermos_size          -- Friday
  | _ => 0

noncomputable def reduced_consumption (day : Nat) : ℝ :=
  match day with
  | 1 | 2 | 4 => (1/4) * original_consumption day  -- Monday, Tuesday, Thursday
  | 3 | 5 => (1/2) * original_consumption day      -- Wednesday, Friday
  | _ => 0

noncomputable def weekly_reduced_consumption : ℝ :=
  (reduced_consumption 1) + (reduced_consumption 2) + (reduced_consumption 3) +
  (reduced_consumption 4) + (reduced_consumption 5)

theorem reduced_weekly_consumption :
  weekly_reduced_consumption = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_weekly_consumption_l794_79429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l794_79483

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.cos t.C = 1/5 ∧ t.a * t.b * Real.cos t.C = 1 ∧ t.a + t.b = Real.sqrt 37

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin (2 * t.C + π/4) = (8 * Real.sqrt 3 - 23 * Real.sqrt 2) / 50 ∧
  t.c = 5 ∧
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l794_79483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_identification_possible_l794_79403

/-- Represents a coin, which can be either genuine or fake -/
inductive Coin
| genuine
| fake
deriving BEq, Repr

/-- Represents the result of a weighing -/
inductive WeighResult
| balanced
| leftHeavier
| rightHeavier

/-- Represents a strategy for identifying genuine coins -/
structure Strategy :=
(weighings : List (List Coin × List Coin))
(payments : List Coin)
(identified : List Coin)

/-- The coin identification problem -/
def coinProblem (n : Nat) (k : Nat) (m : Nat) : Prop :=
∀ (coins : List Coin),
  coins.length = n →
  coins.count Coin.genuine = n - 1 →
  coins.count Coin.fake = 1 →
  ∃ (s : Strategy),
    s.weighings.length + s.payments.length ≤ k ∧
    s.identified.length ≥ m ∧
    s.identified.all (· == Coin.genuine) ∧
    (∀ c, c ∈ s.identified → c ∉ s.payments)

theorem coin_identification_possible : coinProblem 8 3 5 := by
  sorry

#check coin_identification_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_identification_possible_l794_79403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l794_79460

/-- Sequence a_n with given properties -/
def sequence_a : ℕ → ℝ := sorry

/-- Sum of first n terms of sequence a_n -/
def S : ℕ → ℝ := sorry

/-- Theorem stating the properties of sequence a_n and its geometric subsequence -/
theorem sequence_properties :
  (sequence_a 1 = 4) ∧ 
  (∀ n : ℕ, n ≥ 1 → 4 * (S n) = (sequence_a n)^2 + 2 * (sequence_a n) - 8) →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2 * n + 2) ∧
  (∃ k : ℕ → ℕ, k 1 = 1 ∧ ∀ m : ℕ, m ≥ 1 → sequence_a (k m) = 2^(m+1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l794_79460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l794_79476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2^x else x^2 - 6*x + 9

-- State the theorem
theorem f_inequality (x : ℝ) : f x > f 1 ↔ x < 1 ∨ x > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l794_79476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_after_noon_l794_79470

/-- Represents the journey details -/
structure Journey where
  distance : ℚ  -- distance in kilometers
  speed : ℚ     -- speed in kilometers per hour
  start_time : ℚ -- start time in hours (0 represents midnight)

/-- Calculates the arrival time given a journey -/
def arrival_time (j : Journey) : ℚ :=
  j.start_time + j.distance / j.speed

/-- Theorem stating that the arrival time is after noon -/
theorem arrival_after_noon (j : Journey) 
  (h1 : j.distance = 259)
  (h2 : j.speed = 60)
  (h3 : j.start_time = 8) : 
  arrival_time j > 12 := by
  sorry

/-- Compute the arrival time for the given journey -/
def example_journey : ℚ :=
  arrival_time {distance := 259, speed := 60, start_time := 8}

#eval example_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_after_noon_l794_79470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_origin_four_l794_79481

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The x-coordinate of the intersection point of a linear function with the x-axis -/
noncomputable def xIntersection (f : LinearFunction) : ℝ :=
  f.intercept / (-f.slope)

/-- The y-coordinate of the intersection point of a linear function with the x-axis -/
def yIntersection : ℝ := 0

theorem intersection_not_origin_four (f : LinearFunction)
  (h1 : f.slope = -2)
  (h2 : f.intercept = 4) :
  (xIntersection f, yIntersection) ≠ (0, 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_not_origin_four_l794_79481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l794_79419

theorem tan_half_angle (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = -3 / Real.sqrt 5)
  (h2 : abs (Real.sin α) > abs (Real.cos α)) :
  Real.tan (α / 2) = -(Real.sqrt 5 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_l794_79419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l794_79447

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/3
  (∑' n : ℕ, a * r^n) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l794_79447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l794_79417

theorem greatest_integer_fraction : 
  ⌊(5^50 + 3^50 : ℝ) / (5^45 + 3^45 : ℝ)⌋ = 3124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l794_79417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l794_79474

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines an ellipse with given foci and a point P on the ellipse -/
structure Ellipse where
  f1 : Point
  f2 : Point
  p : Point
  h1 : f1 = ⟨0, -1⟩
  h2 : f2 = ⟨0, 1⟩
  h3 : distance f1 f2 = (distance f1 p + distance f2 p) / 2

/-- The theorem stating the equation of the ellipse -/
theorem ellipse_equation (e : Ellipse) : 
  ∀ (x y : ℝ), x^2 / 3 + y^2 / 4 = 1 ↔ ∃ (p : Point), p = ⟨x, y⟩ ∧ 
    distance e.f1 p + distance e.f2 p = distance e.f1 e.f2 * 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l794_79474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l794_79436

theorem roots_of_equation (h : 32 * (Real.cos (6 * π / 180))^5 - 
                               40 * (Real.cos (6 * π / 180))^3 + 
                               10 * (Real.cos (6 * π / 180)) - Real.sqrt 3 = 0) :
  ∀ θ : ℝ, θ ∈ ({78, 150, 222, 294} : Set ℝ) →
    32 * (Real.cos (θ * π / 180))^5 - 
    40 * (Real.cos (θ * π / 180))^3 + 
    10 * (Real.cos (θ * π / 180)) - Real.sqrt 3 = 0 :=
by
  intro θ hθ
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l794_79436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l794_79441

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- The theorem statement -/
theorem triangle_max_area (t : Triangle) 
  (h1 : (t.a - t.b + t.c) / t.c = t.b / (t.a + t.b - t.c))
  (h2 : t.a = 2) :
  ∃ (max_area : ℝ), ∀ (s : Triangle), 
    (s.a - s.b + s.c) / s.c = s.b / (s.a + s.b - s.c) → 
    s.a = 2 → 
    area s ≤ max_area ∧ 
    max_area = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l794_79441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_shortest_parallel_transitive_l794_79437

-- Define a point and a line in a plane
variable (P : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2)))

-- Define a function that gives the length of a line segment from a point to a line
noncomputable def distanceToLine (P : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Define a function that checks if a line segment is perpendicular to a line
def isPerpendicular (P Q : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Theorem 1: The perpendicular segment is the shortest
theorem perpendicular_shortest (P : EuclideanSpace ℝ (Fin 2)) (L : Set (EuclideanSpace ℝ (Fin 2))) :
  ∀ Q : EuclideanSpace ℝ (Fin 2), Q ∈ L → isPerpendicular P Q L → distanceToLine P L ≤ dist P Q := by sorry

-- Define parallel lines
def parallel (L1 L2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define perpendicular lines
def perpendicular (L1 L2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Theorem 2: If a ⊥ b, b ∥ c, c ⊥ d, then a ∥ d
theorem parallel_transitive (a b c d : Set (EuclideanSpace ℝ (Fin 2))) :
  perpendicular a b → parallel b c → perpendicular c d → parallel a d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_shortest_parallel_transitive_l794_79437
