import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_difference_l1318_131802

theorem rationalize_cube_root_difference : 
  ∃ (A B C : ℕ) (D : ℕ+),
    (A = 25 ∧ B = 15 ∧ C = 9 ∧ D = 2) ∧
    (1 / ((5 : ℝ) ^ (1/3) - (3 : ℝ) ^ (1/3)) = 
     ((A : ℝ) ^ (1/3) + (B : ℝ) ^ (1/3) + (C : ℝ) ^ (1/3)) / (D : ℝ)) ∧
    (∀ (k : ℕ+), (k : ℕ) ∣ A ∧ (k : ℕ) ∣ B ∧ (k : ℕ) ∣ C ∧ k ∣ D → k = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_difference_l1318_131802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_l1318_131816

noncomputable section

-- Define the triangle XYZ
def X : ℝ × ℝ := (10, 24)
def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (28, 0)
def A : ℝ × ℝ := (10, 0)

-- Define the conditions
def YZ_length : ℝ := 28
def XA_length : ℝ := 24
def YA_length : ℝ := 10

-- Define the perpendicularity condition
def XA_perpendicular_to_YZ : Prop :=
  (X.1 - A.1) * (Z.1 - Y.1) + (X.2 - A.2) * (Z.2 - Y.2) = 0

-- Define the incenter formula
noncomputable def incenter (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  let a := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let b := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let c := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  ((a*P.1 + b*Q.1 + c*R.1) / (a + b + c), (a*P.2 + b*Q.2 + c*R.2) / (a + b + c))

-- Define M and N
noncomputable def M : ℝ × ℝ := incenter X Y A
noncomputable def N : ℝ × ℝ := incenter X Z A

-- Define the distance formula
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem length_of_MN :
  XA_perpendicular_to_YZ →
  distance M N = 2 * Real.sqrt 26 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_MN_l1318_131816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_line_segment_l1318_131841

/-- Helper function to calculate the area swept by a line segment --/
noncomputable def area_swept_by_segment (A : ℝ × ℝ) (P : ℝ → ℝ × ℝ) (t₁ t₂ : ℝ) : ℝ :=
  sorry

/-- The area swept by a line segment AP as P moves along a circular arc --/
theorem area_swept_by_line_segment (A : ℝ × ℝ) (P : ℝ → ℝ × ℝ) (t₁ t₂ : ℝ) : 
  A = (2, 0) →
  (∀ t, P t = (Real.sin (2 * t - π / 3), Real.cos (2 * t - π / 3))) →
  t₁ = π / 12 →
  t₂ = π / 4 →
  area_swept_by_segment A P t₁ t₂ = π / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_line_segment_l1318_131841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_crossing_bridge_l1318_131895

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 155) 
  (h2 : bridge_length = 220) 
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check train_speed_crossing_bridge

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_crossing_bridge_l1318_131895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_with_AB_diameter_l1318_131873

-- Define the points A and B
def A : ℝ × ℝ := (1, -4)
def B : ℝ × ℝ := (-5, 4)

-- Define the center of the circle
noncomputable def C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the radius of the circle
noncomputable def r : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2

-- Theorem statement
theorem circle_equation_with_AB_diameter :
  ∀ (x y : ℝ), (x + 2)^2 + y^2 = 25 ↔
  (x - C.1)^2 + (y - C.2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_with_AB_diameter_l1318_131873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1318_131893

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (x^2) + 1

-- State the theorem
theorem unique_intersection :
  ∃! x : ℝ, x > 0 ∧ f x = g x :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1318_131893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_marks_l1318_131899

/-- Proves that given the conditions from the problem, the marks in Physics is 95. -/
theorem physics_marks (P C M : ℚ) : 
  (P + C + M) / 3 = 75 →  -- Average of 3 subjects is 75
  (P + M) / 2 = 90 →      -- Average of Physics and Mathematics is 90
  (P + C) / 2 = 70 →      -- Average of Physics and Chemistry is 70
  P = 95 := by
  intro h1 h2 h3
  -- Proof steps would go here
  sorry

#check physics_marks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_physics_marks_l1318_131899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1318_131826

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- One of the asymptotes of the hyperbola -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2 * x

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Main theorem: If one asymptote of the hyperbola is y = 2x, then its eccentricity is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (asymptote : ∀ x y, hyperbola_equation h x y → asymptote_equation x y) :
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1318_131826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lithium_hydroxide_formation_l1318_131851

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  lithium_nitride : Moles
  water : Moles
  lithium_hydroxide : Moles
  ammonia : Moles

/-- The balanced chemical equation for the reaction -/
def balanced_equation (r : ChemicalReaction) : Prop :=
  r.lithium_nitride = (1 : ℝ) ∧ r.water = (3 : ℝ) ∧ r.lithium_hydroxide = (3 : ℝ) ∧ r.ammonia = (1 : ℝ)

/-- The theorem stating that 3 moles of Lithium hydroxide are formed -/
theorem lithium_hydroxide_formation (r : ChemicalReaction) 
  (h : balanced_equation r) : r.lithium_hydroxide = (3 : ℝ) := by
  sorry

-- Instance to allow using natural numbers as Moles
instance : Coe ℕ Moles := ⟨λ n => (n : ℝ)⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lithium_hydroxide_formation_l1318_131851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l1318_131897

/-- Calculates the speed for the remaining part of a trip given the total distance,
    initial distance, initial speed, and average speed for the entire trip. -/
noncomputable def remainingSpeed (totalDistance initialDistance initialSpeed averageSpeed : ℝ) : ℝ :=
  let initialTime := initialDistance / initialSpeed
  let totalTime := totalDistance / averageSpeed
  let remainingTime := totalTime - initialTime
  let remainingDistance := totalDistance - initialDistance
  remainingDistance / remainingTime

/-- Theorem stating that for a 60-mile trip with the first 12 miles at 24 mph
    and an average speed of 40 mph, the speed for the remaining part is 48 mph. -/
theorem trip_speed_calculation :
  remainingSpeed 60 12 24 40 = 48 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l1318_131897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_l1318_131821

/-- Complex number representing rotation by π/6 radians -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * (Real.pi / 6))

/-- Position of the particle after n moves -/
noncomputable def position (n : ℕ) : ℂ := 3 * ω ^ n + 6 * (1 - ω ^ n) / (1 - ω)

/-- The final position after 300 moves is (3, 0) -/
theorem final_position :
  position 300 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_l1318_131821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_op_theorem_l1318_131889

-- Define the custom vector operation
def vector_op (a b : ℤ × ℤ) : ℤ × ℤ :=
  let (m, n) := a
  let (p, q) := b
  (m*p - n*q, m*q + n*p)

-- Define the vector norm
noncomputable def vector_norm (v : ℤ × ℤ) : ℝ :=
  let (x, y) := v
  Real.sqrt (x^2 + y^2 : ℝ)

theorem vector_op_theorem :
  -- Part 1
  vector_op (1, 2) (2, 1) = (0, 5) ∧
  -- Part 2
  ∃ (a b : ℤ × ℤ), 
    vector_op a b = (5, 0) ∧ 
    vector_norm a < 5 ∧ 
    vector_norm b < 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_op_theorem_l1318_131889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_three_l1318_131896

/-- The function representing the relationship between x and y -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2) + x^2

/-- The condition that x is a multiple of y -/
def g (a x : ℝ) : Prop := x = a * f x

/-- Theorem stating that the minimum value of x occurs when a = 3 -/
theorem min_value_at_three :
  ∃ (a : ℝ), ∀ (x : ℝ), x > 2 → g a x → (∀ (b : ℝ), g b x → a ≤ b) ∧ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_at_three_l1318_131896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l1318_131835

/-- Calculates the tax amount based on taxable income --/
noncomputable def calculate_tax (taxable_income : ℝ) : ℝ :=
  if taxable_income ≤ 3000 then taxable_income * 0.03
  else if taxable_income ≤ 12000 then 3000 * 0.03 + (taxable_income - 3000) * 0.10
  else if taxable_income ≤ 25000 then 3000 * 0.03 + 9000 * 0.10 + (taxable_income - 12000) * 0.20
  else 3000 * 0.03 + 9000 * 0.10 + 13000 * 0.20 + (taxable_income - 25000) * 0.25

theorem tax_calculation_correct (gross_income : ℝ) :
  let tax_threshold : ℝ := 5000
  let elderly_support_deduction : ℝ := 1000
  let taxable_income : ℝ := gross_income - tax_threshold - elderly_support_deduction
  let tax_amount : ℝ := calculate_tax taxable_income
  tax_amount = 180 → gross_income - tax_amount = 9720 :=
by
  sorry

-- Remove the #eval line as it might cause issues in this context

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l1318_131835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1318_131842

-- Define the sector and its properties
def sector_radius : ℝ := 9

-- Define the sector angle as a parameter
def sector_angle : ℝ → ℝ := id

-- Define the radius of the circumscribed circle
noncomputable def circumscribed_radius (θ : ℝ) : ℝ := 4.5 * (1 / Real.cos (θ / 2))

-- Theorem statement
theorem sector_circumscribed_circle_radius (θ : ℝ) :
  circumscribed_radius θ = sector_radius / (2 * Real.cos (θ / 2)) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_circumscribed_circle_radius_l1318_131842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_given_ellipse_l1318_131815

/-- Represents an ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The given ellipse with equation x²/4 + y²/3 = 1 -/
noncomputable def given_ellipse : Ellipse where
  a := 2
  b := Real.sqrt 3
  h_positive_a := by norm_num
  h_positive_b := by norm_num

theorem eccentricity_of_given_ellipse :
  eccentricity given_ellipse = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_given_ellipse_l1318_131815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l1318_131834

/-- The height of water in a cylindrical container when water from a cone is poured into it -/
noncomputable def water_height_in_cylinder (cone_height : ℝ) : ℝ :=
  cone_height / 3

theorem water_height_theorem (cone_height : ℝ) (h_positive : cone_height > 0) :
  water_height_in_cylinder cone_height = 3 → cone_height = 9 := by
  intro h
  have : cone_height / 3 = 3 := h
  linarith

#check water_height_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_theorem_l1318_131834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrationals_in_list_l1318_131867

theorem two_irrationals_in_list : ∃ (a b : ℝ), 
  a ≠ b ∧ 
  a ∈ ({4, 0, 12/7, Real.rpow 0.125 (1/3), 0.1010010001, Real.sqrt 3, π/2} : Set ℝ) ∧ 
  b ∈ ({4, 0, 12/7, Real.rpow 0.125 (1/3), 0.1010010001, Real.sqrt 3, π/2} : Set ℝ) ∧ 
  Irrational a ∧ 
  Irrational b ∧
  ∀ c ∈ ({4, 0, 12/7, Real.rpow 0.125 (1/3), 0.1010010001, Real.sqrt 3, π/2} : Set ℝ), 
    Irrational c → (c = a ∨ c = b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_irrationals_in_list_l1318_131867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1318_131863

/-- The polar equation of the circle -/
noncomputable def polar_equation (θ : ℝ) : ℝ := 3 * Real.cos θ - 4 * Real.sin θ

/-- The area of the circle described by the polar equation -/
noncomputable def circle_area : ℝ := 25 * Real.pi / 4

theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (polar_equation θ)^2 = (center.1 + radius * Real.cos θ)^2 + (center.2 + radius * Real.sin θ)^2) ∧
    circle_area = Real.pi * radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_polar_equation_l1318_131863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_product_weight_l1318_131861

/-- Atomic weight of Copper in g/mol -/
noncomputable def Cu_weight : ℝ := 63.55

/-- Atomic weight of Phosphorus in g/mol -/
noncomputable def P_weight : ℝ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
noncomputable def O_weight : ℝ := 16.00

/-- Atomic weight of Carbon in g/mol -/
noncomputable def C_weight : ℝ := 12.01

/-- Molecular weight of Copper(II) phosphate (Cu3(PO4)2) in g/mol -/
noncomputable def Cu3PO42_weight : ℝ := 3 * Cu_weight + 2 * P_weight + 8 * O_weight

/-- Molecular weight of Carbon dioxide (CO2) in g/mol -/
noncomputable def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- Number of moles of Copper(II) carbonate (CuCO3) -/
noncomputable def CuCO3_moles : ℝ := 8

/-- Number of moles of Diphosphorus pentoxide (P4O10) -/
noncomputable def P4O10_moles : ℝ := 6

/-- Stoichiometric ratio of Cu3(PO4)2 to CuCO3 -/
noncomputable def Cu3PO42_ratio : ℝ := 1 / 3

/-- Stoichiometric ratio of CO2 to CuCO3 -/
noncomputable def CO2_ratio : ℝ := 1

theorem total_product_weight : 
  (CuCO3_moles * Cu3PO42_ratio * Cu3PO42_weight) + 
  (CuCO3_moles * CO2_ratio * CO2_weight) = 1368.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_product_weight_l1318_131861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1318_131805

def A : Finset ℕ := {4}
def B : Finset ℕ := {1, 2}
def C : Finset ℕ := {1, 3, 5}

def Point := ℕ × ℕ × ℕ

def validPoints : Finset Point :=
  Finset.product A (Finset.product B C)

theorem distinct_points_count :
  Finset.card validPoints = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_points_count_l1318_131805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_correct_l1318_131887

/-- Represents the time (in hours) it takes to fill the basin for each water source -/
structure FillingTimes where
  rightEye : ℚ
  leftEye : ℚ
  rightFoot : ℚ
  throat : ℚ

/-- Calculates the time to fill the basin when all sources are open simultaneously -/
def simultaneousFillTime (times : FillingTimes) : ℚ :=
  1 / (1 / times.rightEye + 1 / times.leftEye + 1 / times.rightFoot + 1 / times.throat)

/-- The given filling times for each water source -/
def givenTimes : FillingTimes := {
  rightEye := 48,  -- 2 days in hours
  leftEye := 72,   -- 3 days in hours
  rightFoot := 96, -- 4 days in hours
  throat := 6
}

theorem simultaneous_fill_time_correct :
  simultaneousFillTime givenTimes = 4 + 44 / 61 := by
  -- Proof steps would go here
  sorry

#eval simultaneousFillTime givenTimes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_fill_time_correct_l1318_131887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_l1318_131813

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 120 / 97) 
  (h2 : Real.cos x + Real.cos y = 63 / 97) : 
  Real.tan x + Real.tan y = 70656 / 24291 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_l1318_131813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1318_131843

-- Define the functions f and g
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin x * Real.sin (x + 3 * φ)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x - φ)

-- State the theorem
theorem symmetry_of_g (φ : ℝ) :
  (φ ∈ Set.Ioo 0 (Real.pi / 2)) →
  (∀ x, f φ x = -f φ (-x)) →
  (φ = Real.pi / 6 ∧
   ∀ x, g φ (x - 5 * Real.pi / 12) = g φ (-x - 5 * Real.pi / 12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_g_l1318_131843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l1318_131829

theorem determinant_transformation (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = -5 →
  Matrix.det !![x, 3*x + 4*y; z, 3*z + 4*w] = -20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l1318_131829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eighteenth_smallest_in_T_l1318_131881

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2018}

def T : Set ℕ := {n | ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a * b}

noncomputable def nthSmallest (n : ℕ) (S : Set ℕ) : ℕ := sorry

theorem two_thousand_eighteenth_smallest_in_T :
  nthSmallest 2018 T = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_eighteenth_smallest_in_T_l1318_131881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_function_property_l1318_131877

-- Define the set of bijective functions from ℕ+ to ℕ+
def B : Type := {f : ℕ+ → ℕ+ // Function.Bijective f}

-- Define the property that F must satisfy
def SatisfiesProperty (F : B → ℝ) : Prop :=
  ∀ p q : B, (F p + F q)^2 = F (⟨p.1 ∘ p.1, sorry⟩) + F (⟨p.1 ∘ q.1, sorry⟩) + 
                             F (⟨q.1 ∘ p.1, sorry⟩) + F (⟨q.1 ∘ q.1, sorry⟩)

-- Theorem statement
theorem bijective_function_property :
  ∀ F : B → ℝ, SatisfiesProperty F → ∀ p : B, F p = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijective_function_property_l1318_131877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_5_l1318_131857

/-- A line that always bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects_circle : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The minimum distance from (2,2) to the bisecting line -/
noncomputable def min_distance (l : BisectingLine) : ℝ :=
  Real.sqrt ((l.a - 2)^2 + (l.b - 2)^2)

/-- The theorem stating the minimum distance is √5 -/
theorem min_distance_is_sqrt_5 :
  ∃ (l : BisectingLine), ∀ (l' : BisectingLine), min_distance l ≤ min_distance l' ∧ min_distance l = Real.sqrt 5 := by
  sorry

#check min_distance_is_sqrt_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_5_l1318_131857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_hours_worked_l1318_131888

/-- Calculates Harry's pay for a given number of hours worked and hourly rate -/
def harry_pay (hours : ℕ) (x : ℝ) : ℝ :=
  if hours ≤ 24 then x * hours
  else 24 * x + 1.5 * x * (hours - 24)

/-- Calculates James's pay for a given number of hours worked and hourly rate -/
def james_pay (hours : ℕ) (x : ℝ) : ℝ :=
  if hours ≤ 40 then x * hours
  else 40 * x + 2 * x * (hours - 40)

/-- Proves that James worked 41 hours given the conditions -/
theorem james_hours_worked (x : ℝ) (h : x > 0) :
  ∃ j : ℕ, harry_pay 36 x = james_pay j x ∧ j = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_hours_worked_l1318_131888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_half_l1318_131812

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - (b / a) ^ 2)

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse a b) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

/-- The upper vertex of an ellipse -/
def upper_vertex (e : Ellipse a b) : ℝ × ℝ := (0, b)

/-- The lower vertex of an ellipse -/
def lower_vertex (e : Ellipse a b) : ℝ × ℝ := (0, -b)

/-- The symmetric point of a point with respect to a line -/
noncomputable def symmetric_point (p q r : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The radius of a circle passing through three points -/
noncomputable def circle_radius (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity_is_half (a b : ℝ) (e : Ellipse a b) :
  let A := upper_vertex e
  let B := lower_vertex e
  let F := right_focus e
  let B' := symmetric_point B A F
  circle_radius A B' F = a →
  eccentricity e = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_is_half_l1318_131812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rosette_area_correct_l1318_131823

/-- The area of a rosette formed by semicircles on each side of a square -/
noncomputable def rosette_area (a : ℝ) : ℝ := (a^2 * (Real.pi - 2)) / 2

/-- Theorem stating that the area of the rosette is correct -/
theorem rosette_area_correct (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_diameter := a
  rosette_area a = (a^2 * (Real.pi - 2)) / 2 := by
  -- Unfold the definition of rosette_area
  unfold rosette_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rosette_area_correct_l1318_131823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_area_is_four_sqrt_three_thirds_l1318_131831

/-- Configuration of a regular pentagon surrounded by equilateral triangles -/
structure PentagonTriangleConfig where
  /-- Side length of the regular pentagon -/
  pentagon_side : ℝ
  /-- Side length of each equilateral triangle -/
  triangle_side : ℝ
  /-- The pentagon is regular with side length 2 -/
  pentagon_regular : pentagon_side = 2
  /-- Each triangle is equilateral with side length 2 -/
  triangle_equilateral : triangle_side = 2
  /-- There are five triangles, each sharing a side with the pentagon -/
  triangle_count : Fin 5

/-- The area of the triangle formed by connecting the centers of three adjacent triangles -/
noncomputable def center_triangle_area (config : PentagonTriangleConfig) : ℝ :=
  4 * Real.sqrt 3 / 3

/-- Theorem stating the area of the triangle formed by connecting the centers of three adjacent triangles -/
theorem center_triangle_area_is_four_sqrt_three_thirds (config : PentagonTriangleConfig) :
  center_triangle_area config = 4 * Real.sqrt 3 / 3 := by
  sorry

#check center_triangle_area_is_four_sqrt_three_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_triangle_area_is_four_sqrt_three_thirds_l1318_131831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lilly_buys_eleven_flowers_l1318_131872

/-- The number of flowers Lilly can buy for Maria's birthday -/
def flowers_for_maria (days : ℕ) (savings_per_day : ℚ) (flower_cost : ℚ) : ℕ :=
  (days : ℚ) * savings_per_day / flower_cost |>.floor.toNat

/-- Theorem: Lilly can buy 11 flowers for Maria -/
theorem lilly_buys_eleven_flowers :
  flowers_for_maria 22 2 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lilly_buys_eleven_flowers_l1318_131872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_l1318_131825

/-- The average number of people infected by one person in each round -/
def infection_rate : ℕ → ℕ := sorry

/-- The total number of people with flu after n rounds -/
def total_infected : ℕ → ℕ := sorry

theorem flu_spread (h : total_infected 2 = 81) :
  infection_rate 1 = 8 ∧ total_infected 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_l1318_131825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1318_131804

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_slope : (b / a) = (4 / 3)) : 
  Real.sqrt ((a^2 + b^2) / a^2) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1318_131804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_conditions_satisfied_by_2493_smallest_number_is_2493_l1318_131883

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_two_even_two_odd_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 2 ∧ (digits.filter (λ d => d % 2 = 1)).length = 2

def thousands_digit_at_least_two (n : ℕ) : Prop :=
  (n / 1000) ≥ 2

theorem smallest_four_digit_number (n : ℕ) :
  (is_four_digit n ∧
   n % 9 = 0 ∧
   has_two_even_two_odd_digits n ∧
   thousands_digit_at_least_two n) →
  n ≥ 2493 :=
by sorry

theorem conditions_satisfied_by_2493 :
  is_four_digit 2493 ∧
  2493 % 9 = 0 ∧
  has_two_even_two_odd_digits 2493 ∧
  thousands_digit_at_least_two 2493 :=
by sorry

theorem smallest_number_is_2493 :
  ∀ n : ℕ, (is_four_digit n ∧
            n % 9 = 0 ∧
            has_two_even_two_odd_digits n ∧
            thousands_digit_at_least_two n) →
  n = 2493 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_conditions_satisfied_by_2493_smallest_number_is_2493_l1318_131883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_cost_theorem_l1318_131800

/-- Represents the dimensions and cost parameters of a rectangular plot with a gravel path --/
structure PlotWithPath where
  length : ℝ
  width : ℝ
  path_width : ℝ
  cost_per_sqm_paise : ℝ

/-- Calculates the cost of gravelling the path for a given plot --/
noncomputable def gravelling_cost (plot : PlotWithPath) : ℝ :=
  let total_length := plot.length + 2 * plot.path_width
  let total_width := plot.width + 2 * plot.path_width
  let total_area := total_length * total_width
  let inner_area := plot.length * plot.width
  let path_area := total_area - inner_area
  let cost_per_sqm_rupees := plot.cost_per_sqm_paise / 100
  path_area * cost_per_sqm_rupees

/-- Theorem stating that the cost of gravelling the path for the given plot is 1270.5 rupees --/
theorem gravelling_cost_theorem (plot : PlotWithPath) 
    (h1 : plot.length = 150)
    (h2 : plot.width = 85)
    (h3 : plot.path_width = 3.5)
    (h4 : plot.cost_per_sqm_paise = 75) : 
  gravelling_cost plot = 1270.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravelling_cost_theorem_l1318_131800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_day_distance_l1318_131885

/-- Represents the distance traveled each day -/
noncomputable def distance_sequence (first_day_distance : ℝ) : ℕ → ℝ
  | 0 => first_day_distance
  | n + 1 => (distance_sequence first_day_distance n) / 2

/-- Calculates the sum of distances for a given number of days -/
noncomputable def total_distance (first_day_distance : ℝ) (days : ℕ) : ℝ :=
  (List.range days).map (distance_sequence first_day_distance) |>.sum

theorem last_day_distance (first_day_distance : ℝ) :
  (total_distance first_day_distance 6 = 378) →
  (distance_sequence first_day_distance 5 = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_day_distance_l1318_131885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_l1318_131838

/-- Given four points A, B, C, and D on a line with AB = 1, BC = 2, and CD = 4,
    the possible values for AD are 1, 3, 5, and 7. -/
theorem possible_distances (A B C D : ℝ) : 
  (∃ (order : List ℝ), order.Perm [A, B, C, D]) →
  |B - A| = 1 →
  |C - B| = 2 →
  |D - C| = 4 →
  |D - A| ∈ ({1, 3, 5, 7} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_l1318_131838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3_of_expansion_l1318_131882

-- Define the expansion
noncomputable def expansion (x : ℝ) : ℝ := x * (1 + Real.sqrt x)^6

-- Define the coefficient of x^3 in the expansion
noncomputable def coefficient_x3 (f : ℝ → ℝ) : ℝ :=
  (deriv (deriv (deriv f))) 0 / 6

-- Theorem statement
theorem coefficient_x3_of_expansion :
  coefficient_x3 expansion = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3_of_expansion_l1318_131882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_multiple_is_integer_l1318_131848

/-- A real polynomial of degree n that takes integer values for integer inputs -/
def IntegerValuedPolynomial (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n ∧ ∀ m : ℤ, ∃ k : ℤ, p.eval (m : ℝ) = k

/-- The main theorem: if p is an integer-valued polynomial of degree n,
    then n! times any coefficient of p is an integer -/
theorem coefficient_multiple_is_integer
    (p : Polynomial ℝ) (n : ℕ) (h : IntegerValuedPolynomial p n) :
    ∀ i : ℕ, ∃ z : ℤ, (n.factorial : ℝ) * p.coeff i = z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_multiple_is_integer_l1318_131848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l1318_131868

/-- Given a real number a and a function f(x) = e^x + a * e^(-x) whose second derivative
    is an odd function, prove that the abscissa of the point where the tangent line
    has a slope of 3/2 is ln 2. -/
theorem tangent_point_abscissa (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.exp (-x)
  let f' := deriv f
  let f'' := deriv f'
  (∀ x, f'' (-x) = -(f'' x)) →  -- f'' is an odd function
  ∃ x, f' x = 3/2 ∧ x = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_abscissa_l1318_131868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_quartic_equation_l1318_131811

theorem solution_set_of_quartic_equation :
  let S : Set ℂ := {z : ℂ | z^4 - 6*z^2 + 8 = 0}
  S = {-2, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_quartic_equation_l1318_131811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_12_l1318_131820

def is_divisor_of_12 (n : ℕ) : Bool := 12 % n = 0

def fair_6_sided_die : Finset ℕ := Finset.range 6

theorem probability_divisor_of_12 :
  (Finset.filter (fun n => is_divisor_of_12 n) fair_6_sided_die).card / fair_6_sided_die.card = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_12_l1318_131820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_should_play_B_first_l1318_131807

/-- Represents a player in the game series -/
inductive Player : Type
| A : Player
| B : Player
| C : Player

/-- The probability of one player winning against another -/
noncomputable def win_prob (p1 p2 : Player) : ℝ := sorry

/-- A is the weakest player, C is the strongest -/
axiom player_strength (p1 p2 : Player) :
  (p1 = Player.A ∧ p2 ≠ Player.A) → win_prob p1 p2 < win_prob p2 p1

/-- The probability of A winning the series when A plays against B first -/
noncomputable def prob_A_wins_AB_first : ℝ := sorry

/-- The probability of A winning the series when B plays against C first -/
noncomputable def prob_A_wins_BC_first : ℝ := sorry

/-- A should choose to play against B first -/
theorem A_should_play_B_first :
  prob_A_wins_AB_first > prob_A_wins_BC_first := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_should_play_B_first_l1318_131807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_term_l1318_131886

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 1
  else sequence_a (n - 1) - 1 / ((n : ℚ) * (n + 1 : ℚ)) + 1

theorem sequence_tenth_term :
  sequence_a 9 = 91 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_tenth_term_l1318_131886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_sine_l1318_131869

theorem max_omega_for_increasing_sine (A ω : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x y : ℝ, x ∈ Set.Icc (-π/2) (2*π/3) → y ∈ Set.Icc (-π/2) (2*π/3) → 
    x < y → A * Real.sin (ω * x) < A * Real.sin (ω * y)) →
  ω ≤ 3/4 ∧ ∃ ω₀ : ℝ, ω₀ = 3/4 ∧ 
    (∀ x y : ℝ, x ∈ Set.Icc (-π/2) (2*π/3) → y ∈ Set.Icc (-π/2) (2*π/3) → 
      x < y → A * Real.sin (ω₀ * x) < A * Real.sin (ω₀ * y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_for_increasing_sine_l1318_131869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l1318_131846

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 13

-- Define a point being on a curve
def on_curve (P : ℝ × ℝ) (curve : ℝ → ℝ → Prop) : Prop := curve P.1 P.2

-- Define a line being tangent to a curve
def is_tangent (l : ℝ → ℝ → Prop) (curve : ℝ → ℝ → Prop) : Prop := sorry

-- Define symmetry with respect to the origin
def symmetric_to_origin (A B : ℝ × ℝ) : Prop := B.1 = -A.1 ∧ B.2 = -A.2

theorem ellipse_and_tangent_line 
  (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a_gt_b : a > b) 
  (h_focus : ellipse a b (Real.sqrt 5) 0)
  (h_eccentricity : Real.sqrt 5 / a = Real.sqrt 5 / 3)
  (P : ℝ × ℝ)
  (h_P_on_M : on_curve P circle_M)
  (A : ℝ × ℝ)
  (h_A_on_M : on_curve A circle_M)
  (B : ℝ × ℝ)
  (h_B_symmetric : symmetric_to_origin A B)
  (l : ℝ → ℝ → Prop)
  (h_l_through_P : l P.1 P.2)
  (h_l_tangent_C : is_tangent l (ellipse a b)) :
  (a = 3 ∧ b = 2) ∧
  is_tangent (λ x y ↦ (y - P.2) = -(x - P.1) * ((B.2 - P.2)/(B.1 - P.1))) (ellipse 3 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l1318_131846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1318_131806

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Circle function with center (3, -r) and radius r -/
def circleEq (x y r : ℝ) : Prop := (x - 3)^2 + (y + r)^2 = r^2

/-- The circle is inscribed if it touches the parabola at exactly one point -/
def is_inscribed (r : ℝ) : Prop :=
  ∃! x, circleEq x (parabola x) r ∧ parabola x ≥ 0

theorem inscribed_circle_radius :
  ∃! r, r > 0 ∧ is_inscribed r ∧ r = 1 := by
  sorry

#check inscribed_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1318_131806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_growth_l1318_131862

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest integer --/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem bank_deposit_growth :
  let principal : ℝ := 1000000
  let rate : ℝ := 0.055
  let time : ℕ := 3
  round_to_nearest (compound_interest principal rate time) = 1174241 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_growth_l1318_131862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_inequality_triangle_ratio_sum_minimum_l1318_131810

/-- Triangle with side lengths a, b, c and area P -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ

/-- Point inside a triangle with perpendicular distances to sides -/
structure PointInTriangle where
  PD : ℝ
  PE : ℝ
  PF : ℝ

/-- The sum of ratios of side lengths to perpendicular distances -/
noncomputable def ratioSum (t : Triangle) (p : PointInTriangle) : ℝ :=
  t.a / p.PD + t.b / p.PE + t.c / p.PF

theorem triangle_ratio_sum_inequality (t : Triangle) (p : PointInTriangle) :
  ratioSum t p ≥ (t.a + t.b + t.c)^2 / (2 * t.P) := by
  sorry

theorem triangle_ratio_sum_minimum (t : Triangle) :
  ∃ p : PointInTriangle, ∀ q : PointInTriangle, ratioSum t p ≤ ratioSum t q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sum_inequality_triangle_ratio_sum_minimum_l1318_131810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1318_131871

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and plane
def Line (V : Type*) [NormedAddCommGroup V] := V × V
def Plane (V : Type*) [NormedAddCommGroup V] := V × V × V

-- Define the perpendicular and subset relations
def perp (l : Line V) (p : Plane V) : Prop := sorry
def perp_line (l1 l2 : Line V) : Prop := sorry
def subset_plane (l : Line V) (p : Plane V) : Prop := sorry

-- State the theorem
theorem perpendicular_necessary_not_sufficient
  (m n : Line V) (α : Plane V) :
  perp m α →
  (subset_plane n α → perp_line m n) ∧
  ¬(perp_line m n → subset_plane n α) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1318_131871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_policeman_always_catches_l1318_131814

/-- Represents a corridor in the problem setup -/
structure Corridor where
  length : ℝ
deriving Inhabited

/-- Represents the setup of the problem -/
structure CatchProblem where
  corridors : Finset Corridor
  policeman_speed : ℝ
  gangster_speed : ℝ
  visibility_range : ℝ

/-- Predicate to check if the policeman can catch the gangster -/
def can_catch (problem : CatchProblem) : Prop :=
  problem.corridors.card = 3 ∧
  ∀ c ∈ problem.corridors, c.length = problem.corridors.toList.head!.length ∧
  problem.policeman_speed = 2 * problem.gangster_speed

/-- Represents a strategy for the policeman to catch the gangster -/
structure Strategy where
  execute : CatchProblem → Bool

/-- Predicate to check if a strategy succeeds -/
def strategy_succeeds (strategy : Strategy) (problem : CatchProblem) : Prop :=
  strategy.execute problem = true

/-- Theorem stating that the policeman can always catch the gangster under the given conditions -/
theorem policeman_always_catches (problem : CatchProblem) :
  can_catch problem → ∃ strategy : Strategy, strategy_succeeds strategy problem :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_policeman_always_catches_l1318_131814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_cubes_l1318_131845

theorem divisibility_of_sum_of_cubes (a b x y : ℤ) :
  ∃ P : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = (a + b) * (x + y) * P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sum_of_cubes_l1318_131845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_identity_l1318_131818

open Real

theorem trigonometric_sum_identity (α β γ : ℝ) 
  (h1 : sin α + sin β + sin γ = 0) 
  (h2 : cos α + cos β + cos γ = 0) : 
  (cos (3 * α) + cos (3 * β) + cos (3 * γ) = 3 * cos (α + β + γ)) ∧ 
  (sin (3 * α) + sin (3 * β) + sin (3 * γ) = 3 * sin (α + β + γ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_identity_l1318_131818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exponents_sqrt_largest_square_15_factorial_l1318_131860

-- Define 15!
def factorial_15 : ℕ := (Finset.range 15).prod (λ i ↦ i + 1)

-- Define the function to calculate the exponent of a prime p in n!
def prime_exp_in_factorial (p n : ℕ) : ℕ :=
  (Finset.range (n / p + 1)).sum (λ k ↦ n / (p ^ (k + 1)))

-- Define the function to get the largest even number not exceeding n
def largest_even_le (n : ℕ) : ℕ :=
  if n % 2 = 0 then n else n - 1

-- Define the function to calculate the sum of exponents in the square root
-- of the largest perfect square dividing n!
def sum_exponents_sqrt_largest_square (n : ℕ) : ℕ :=
  let primes := (Finset.range n).filter Nat.Prime
  (primes.sum (λ p ↦ largest_even_le (prime_exp_in_factorial p n) / 2))

-- State the theorem
theorem sum_exponents_sqrt_largest_square_15_factorial :
  sum_exponents_sqrt_largest_square 15 = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exponents_sqrt_largest_square_15_factorial_l1318_131860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l1318_131876

noncomputable def rent : ℝ := 5000
noncomputable def milk : ℝ := 1500
noncomputable def groceries : ℝ := 4500
noncomputable def education : ℝ := 2500
noncomputable def petrol : ℝ := 2000
noncomputable def miscellaneous : ℝ := 2500

noncomputable def total_expenses : ℝ := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def savings_percentage : ℝ := 0.1

noncomputable def monthly_salary : ℝ := total_expenses / (1 - savings_percentage)
noncomputable def savings : ℝ := monthly_salary * savings_percentage

theorem kishore_savings : 
  ∀ ε > 0, |savings - 2333.33| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kishore_savings_l1318_131876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_d_values_l1318_131879

theorem sum_of_d_values : 
  ∃ d₁ d₂ : ℝ, 
    (d₁^2 + (-d₁)^2 = (12 - d₁)^2 + (2*d₁ - 6)^2) ∧
    (d₂^2 + (-d₂)^2 = (12 - d₂)^2 + (2*d₂ - 6)^2) ∧
    d₁^2 - 16*d₁ + 60 = 0 ∧ 
    d₂^2 - 16*d₂ + 60 = 0 ∧ 
    d₁ + d₂ = 16 := by
  sorry

#check sum_of_d_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_d_values_l1318_131879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_formula_l1318_131853

/-- A regular triangular pyramid with a cube inscribed such that one face of the cube
    lies in the base plane of the pyramid and the remaining four vertices of the cube
    are on the lateral surface of the pyramid. -/
structure PyramidWithCube where
  /-- Side length of the base of the pyramid -/
  a : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Assumption that a and h are positive -/
  a_pos : 0 < a
  h_pos : 0 < h

/-- The edge length of the inscribed cube in the pyramid -/
noncomputable def cubeEdgeLength (p : PyramidWithCube) : ℝ :=
  (3 * p.a) / ((Real.sqrt 3 + 2) * p.h + 3 * p.a)

/-- Theorem stating that the edge length of the inscribed cube is as calculated -/
theorem cube_edge_length_formula (p : PyramidWithCube) :
  ∃ (x : ℝ), x > 0 ∧ x = cubeEdgeLength p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_formula_l1318_131853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approx_0112_l1318_131837

noncomputable def total_smartphones : ℕ := 250
noncomputable def defective_smartphones : ℕ := 84

noncomputable def probability_two_defective : ℝ :=
  (defective_smartphones : ℝ) / (total_smartphones : ℝ) *
  ((defective_smartphones - 1) : ℝ) / ((total_smartphones - 1) : ℝ)

theorem probability_approx_0112 :
  abs (probability_two_defective - 0.112) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approx_0112_l1318_131837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_is_90_75_l1318_131898

def quiz_scores : List Float := [91, 90, 92, 87.5, 89.3, 94.7]

theorem average_score_is_90_75 : 
  (quiz_scores.sum / quiz_scores.length.toFloat : Float) = 90.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_is_90_75_l1318_131898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_l1318_131849

noncomputable section

open Real

-- Define the original function
def g (x : ℝ) : ℝ := sin (2 * x + π / 6)

-- Define f as a shift of g
def f (x : ℝ) : ℝ := g (x + π / 6)

-- Theorem to prove
theorem f_at_pi_third : f (π / 3) = -1 / 2 := by
  -- Expand the definition of f
  unfold f
  -- Expand the definition of g
  unfold g
  -- Simplify the expression
  simp [sin_add, sin_pi_div_two]
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_third_l1318_131849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_not_42_l1318_131817

theorem fraction_sum_not_42 (n m : ℕ+) 
  (h : ∃ k : ℤ, (1 / 2 : ℚ) + (1 / 3 : ℚ) + (1 / 7 : ℚ) + (1 / n.val : ℚ) + (1 / m.val : ℚ) = k) :
  ¬(n = 42 ∧ m = 42) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_not_42_l1318_131817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_interior_points_l1318_131856

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is inside a triangle -/
noncomputable def inside_triangle (p a b c : Point) : Prop :=
  -- Simplified condition; actual implementation would be more complex
  sorry

theorem existence_of_interior_points {n : ℕ} (S : Finset Point) 
  (h1 : S.card = n)
  (h2 : n ≥ 3)
  (h3 : ∀ (p q r : Point), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r) :
  ∃ (M : Finset Point), M.card = 2*n - 5 ∧ 
  ∀ (a b c : Point), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → 
  ∃ p ∈ M, inside_triangle p a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_interior_points_l1318_131856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l1318_131865

/-- Definition of a triangle -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  area : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  inscribedCircleRadius : ℝ

/-- Definition of an isosceles triangle -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- The area of an isosceles triangle with a 120° angle and inscribed circle radius of ∜12 cm -/
theorem isosceles_triangle_area (t : Triangle) (r : ℝ) (h1 : t.isIsosceles) 
  (h2 : t.angle1 = 2 * Real.pi / 3) (h3 : r = (12 : ℝ)^(1/4)) 
  (h4 : t.inscribedCircleRadius = r) : 
  t.area = 2 * (7 + 4 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l1318_131865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l1318_131808

noncomputable def average_speed_by_distance (s1 s2 : ℝ) : ℝ := 4 / (1 / s1 + 1 / s2)

noncomputable def average_speed_by_time (s1 s2 : ℝ) : ℝ := (s1 + s2) / 2

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  average_speed_by_distance (2 * u) (2 * v) ≤ average_speed_by_time (3 * u) (3 * v) := by
  sorry

#check car_speed_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l1318_131808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unit_triangles_in_equilateral_l1318_131892

/-- Checks if two unit triangles are non-overlapping based on their placements -/
def are_non_overlapping (p1 p2 : ℝ × ℝ) : Prop := sorry

/-- Checks if a unit triangle is inside the larger triangle of side length L -/
def is_inside_triangle (L : ℝ) (p : ℝ × ℝ) : Prop := sorry

/-- Checks if a triangle placement represents a unit equilateral triangle -/
def is_unit_triangle (p : ℝ × ℝ) : Prop := sorry

/-- Checks if a unit triangle has sides parallel but opposite orientation to the larger triangle -/
def is_parallel_opposite_orientation (L : ℝ) (p : ℝ × ℝ) : Prop := sorry

/-- Given an equilateral triangle with side length L > 0, the maximum number of
    non-overlapping unit equilateral triangles that can be placed inside it,
    with sides parallel but opposite orientation to the larger triangle,
    is less than or equal to (2/3) * L^2. -/
theorem max_unit_triangles_in_equilateral (L : ℝ) (h : L > 0) :
  ∃ (n : ℕ), n ≤ (2/3 : ℝ) * L^2 ∧
  (∀ (m : ℕ), m > n → ¬ ∃ (placement : ℕ → ℝ × ℝ),
    (∀ i j, i ≠ j → are_non_overlapping (placement i) (placement j)) ∧
    (∀ i, is_inside_triangle L (placement i)) ∧
    (∀ i, is_unit_triangle (placement i)) ∧
    (∀ i, is_parallel_opposite_orientation L (placement i))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unit_triangles_in_equilateral_l1318_131892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1318_131824

theorem beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = Real.sqrt 5 / 5) (h4 : Real.sin (β - α) = -(Real.sqrt 10) / 10) :
  β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1318_131824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_original_number_l1318_131803

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The original number to be rounded -/
def original_number : ℝ := 12345.64789

/-- Theorem stating that rounding the original number to the nearest tenth equals 12345.6 -/
theorem round_to_nearest_tenth_of_original_number :
  round_to_nearest_tenth original_number = 12345.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_of_original_number_l1318_131803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1318_131833

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (3 + 2 * Real.cos θ, -4 + 2 * Real.sin θ)

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the area of triangle ABM
noncomputable def triangle_area (M : ℝ × ℝ) : ℝ :=
  let (x, y) := M
  |2 * x - 2 * y + 9| / 2

theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 9 + 2 * Real.sqrt 2 ∧
  ∀ θ : ℝ, triangle_area (circle_C θ) ≤ max_area := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1318_131833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1318_131890

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 5 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan (-m)

-- State the theorem
theorem line_inclination_angle :
  ∃ (θ : ℝ), 
    (0 ≤ θ ∧ θ < Real.pi) ∧
    inclination_angle (1 / Real.sqrt 3) = θ ∧
    θ = 5 * Real.pi / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1318_131890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_unique_sums_l1318_131850

/-- Represents a triangular grid of small equilateral triangles -/
structure TriangularGrid :=
  (size : Nat)
  (filled : Fin size → Fin 9)

/-- Represents a "3-triangle" in the grid -/
structure ThreeTriangle (n : Nat) :=
  (a b c : Fin n)
  (adjacent : a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- The number of distinct "3-triangles" in a grid of 16 small triangles -/
def num_three_triangles : Nat := 27

/-- The number of possible distinct sums for "3-triangles" -/
def num_possible_sums : Nat := 25

/-- Theorem stating the impossibility of filling the grid with unique sums for all "3-triangles" -/
theorem impossible_unique_sums (grid : TriangularGrid) 
  (h1 : grid.size = 16) 
  (h2 : ∀ t : ThreeTriangle grid.size, 
    (grid.filled t.a).val + (grid.filled t.b).val + (grid.filled t.c).val ≤ 27) 
  (h3 : ∀ t : ThreeTriangle grid.size, 
    (grid.filled t.a).val + (grid.filled t.b).val + (grid.filled t.c).val ≥ 3) :
  ¬(∀ t1 t2 : ThreeTriangle grid.size, t1 ≠ t2 → 
    (grid.filled t1.a).val + (grid.filled t1.b).val + (grid.filled t1.c).val ≠ 
    (grid.filled t2.a).val + (grid.filled t2.b).val + (grid.filled t2.c).val) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_unique_sums_l1318_131850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l1318_131852

noncomputable section

/-- Parabola C₁ -/
def C₁ (p : ℝ) (x y : ℝ) : Prop := y = (1 / (2 * p)) * x^2 ∧ p > 0

/-- Hyperbola C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 / 8 - y^2 = 1

/-- Focus of C₁ -/
def focus_C₁ (p : ℝ) : ℝ × ℝ := (0, p / 2)

/-- Right focus of C₂ -/
def focus_C₂ : ℝ × ℝ := (3, 0)

/-- Line connecting the foci -/
def connecting_line (p : ℝ) (x y : ℝ) : Prop :=
  (p / 2) * x + 3 * y - (3 / 2) * p = 0

/-- Point M on C₁ -/
def point_M (p : ℝ) : ℝ × ℝ := (Real.sqrt 2 / 4 * p, p / 16)

/-- Slope of tangent line at M is parallel to asymptote of C₂ -/
def tangent_parallel_asymptote (p : ℝ) : Prop :=
  (Real.sqrt 2 / 4 * p) / p = Real.sqrt 2 / 4

theorem parabola_hyperbola_intersection (p : ℝ) :
  C₁ p (point_M p).1 (point_M p).2 ∧
  C₂ (focus_C₂.1) (focus_C₂.2) ∧
  connecting_line p (point_M p).1 (point_M p).2 ∧
  tangent_parallel_asymptote p →
  p = 21 * Real.sqrt 2 / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l1318_131852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_share_is_12000_l1318_131884

/-- Represents the profit share of a partner -/
structure PartnerShare where
  parts : ℕ
  amount : ℕ

/-- Calculates the largest share among partners given a total profit and ratio -/
def largestShare (totalProfit : ℕ) (ratio : List ℕ) : ℕ :=
  let totalParts := ratio.sum
  let partValue := totalProfit / totalParts
  (ratio.map (fun p => p * partValue)).maximum?
    |>.getD 0  -- Default to 0 if the list is empty

/-- Theorem: The largest share for the given problem is $12,000 -/
theorem largest_share_is_12000 :
  largestShare 38000 [2, 3, 4, 4, 6] = 12000 := by
  sorry

#eval largestShare 38000 [2, 3, 4, 4, 6]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_share_is_12000_l1318_131884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sine_relations_l1318_131847

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem triangle_cosine_sine_relations (t : Triangle) :
  (Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 = 1 - 2 * Real.cos t.A * Real.cos t.B * Real.cos t.C) ∧
  ((Real.cos t.A / 39 = Real.cos t.B / 33) ∧ (Real.cos t.B / 33 = Real.cos t.C / 25) →
   ∃ (k : Real), k > 0 ∧ Real.sin t.A = 13 * k ∧ Real.sin t.B = 14 * k ∧ Real.sin t.C = 15 * k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sine_relations_l1318_131847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimal_and_twelve_l1318_131839

theorem product_of_repeating_decimal_and_twelve : (1 : ℚ) / 3 * 12 = 4 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_repeating_decimal_and_twelve_l1318_131839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_coordinates_l1318_131809

/-- Given a triangle ABC with known points A, M (midpoint of AB), and L (on angle bisector BL),
    prove that the coordinates of C are as specified. -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) : 
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  M = ((A.1 + 6) / 2, (A.2 + 14) / 2) →  -- M is midpoint of AB, where B = (6, 14)
  (∃ t : ℝ, L = (1 - t) • A + t • (6, 14)) →  -- L is on line AB
  (∃ k : ℝ, (14, 2) = (1 - k) • A + k • L) →  -- C is on line AL
  (14, 2) = ((3 * 14 - 2) / 2, (-14 + 18) / 2) →  -- C satisfies symmetry condition
  ∃ C : ℝ × ℝ, C = (14, 2) ∧ 
    C.1 = (A.1 + 2 * M.1) / 3 ∧
    C.2 = (A.2 + 2 * M.2) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_coordinates_l1318_131809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l1318_131822

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem that f is an increasing function
theorem f_is_increasing : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x < y → f x < f y :=
by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l1318_131822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_possible_values_l1318_131864

noncomputable section

variable (f g : ℝ → ℝ)
variable (a : ℝ)

axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_g_sum : ∀ x, f x + g x = a * x^2 + x + 2
axiom g_monotone : ∀ x₁ x₂, 1 < x₁ → x₁ < x₂ → x₂ < 2 → (g x₁ - g x₂) / (x₁ - x₂) > -2

theorem a_possible_values : a ≥ -1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_possible_values_l1318_131864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_first_term_equality_general_term_formula_l1318_131801

def S (n : ℕ) : ℤ := 3 * n^2 - n + 1

def a : ℕ → ℤ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | n+2 => 6 * (n+2) - 4

theorem sequence_general_term (n : ℕ) :
  n > 0 → S n - S (n-1) = a n := by
  sorry

-- Additional theorem to show that S 1 = a 1
theorem first_term_equality : S 1 = a 1 := by
  sorry

-- Additional theorem to show that the formula works for n ≥ 2
theorem general_term_formula (n : ℕ) :
  n ≥ 2 → a n = 6 * n - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_first_term_equality_general_term_formula_l1318_131801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1318_131891

-- Define the function f
noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x

-- Theorem stating that the maximum value of f is 2
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1318_131891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l1318_131819

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the cowboy -/
theorem cowboy_shortest_path (start cabin : Point) (river_y : ℝ) : 
  start.x = 0 ∧ start.y = -3 ∧
  cabin.x = 10 ∧ cabin.y = 6 ∧
  river_y = 3 →
  ∃ (river_point : Point), 
    river_point.y = river_y ∧
    (river_y - start.y) + distance river_point cabin = 3 + Real.sqrt 109 := by
  sorry

#check cowboy_shortest_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l1318_131819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_difference_is_four_l1318_131875

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 : Type := Fin 3 → Fin 3 → Int

/-- Creates the initial matrix of consecutive calendar dates starting from 8 -/
def initialMatrix : Matrix3x3 :=
  λ i j => 8 + 3 * i.val + j.val

/-- Modifies the first row by adding 3 to each element -/
def modifyFirstRow (m : Matrix3x3) : Matrix3x3 :=
  λ i j => if i = 0 then m i j + 3 else m i j

/-- Reverses the order of the numbers in the first row -/
def reverseFirstRow (m : Matrix3x3) : Matrix3x3 :=
  λ i j => if i = 0 then m i (2 - j) else m i j

/-- Calculates the sum of the main diagonal -/
def mainDiagonalSum (m : Matrix3x3) : Int :=
  m 0 0 + m 1 1 + m 2 2

/-- Calculates the sum of the anti-diagonal -/
def antiDiagonalSum (m : Matrix3x3) : Int :=
  m 0 2 + m 1 1 + m 2 0

/-- The final theorem to prove -/
theorem diagonal_difference_is_four :
  let m := reverseFirstRow (modifyFirstRow initialMatrix)
  (mainDiagonalSum m - antiDiagonalSum m).natAbs = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_difference_is_four_l1318_131875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_polar_axis_l1318_131874

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The polar axis -/
def polarAxis : PolarLine := { equation := λ ρ θ => θ = 0 ∨ θ = Real.pi }

theorem line_parallel_to_polar_axis (a : ℝ) (h : a > 0) :
  let l : PolarLine := { equation := λ ρ θ => ρ * Real.sin θ = a }
  (∃ (p : PolarPoint), p.r = a ∧ l.equation p.r p.θ) ∧
  (∀ (ρ θ : ℝ), l.equation ρ θ → (∃ (k : ℝ), ρ * Real.cos θ = k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_polar_axis_l1318_131874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l1318_131870

def P : Set ℝ := {x | x^2 - 9 < 0}

def Q : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ -1 ≤ n ∧ n ≤ 3}

theorem intersection_P_Q : P ∩ Q = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l1318_131870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sampling_survey_correct_l1318_131894

-- Define the options
inductive OptionChoice
  | A
  | B
  | C
  | D

-- Define the properties of each option
def is_sampling_survey (o : OptionChoice) : Prop :=
  o = OptionChoice.A

def is_lottery_guarantee (o : OptionChoice) : Prop :=
  o = OptionChoice.B

def is_data_comparison (o : OptionChoice) : Prop :=
  o = OptionChoice.C

def is_dice_event (o : OptionChoice) : Prop :=
  o = OptionChoice.D

-- Define the correctness of each option
def is_correct (o : OptionChoice) : Prop :=
  match o with
  | OptionChoice.A => True
  | OptionChoice.B => False
  | OptionChoice.C => False
  | OptionChoice.D => False

-- Theorem statement
theorem only_sampling_survey_correct :
  ∀ o : OptionChoice,
    is_correct o ↔ is_sampling_survey o :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sampling_survey_correct_l1318_131894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_3_graph_translation_cos_2x_plus_pi_3_is_left_translation_of_sin_2x_l1318_131827

open Real

theorem cos_2x_plus_pi_3_graph_translation (x : ℝ) :
  cos (2 * x + π / 3) = sin (2 * (x + 5 * π / 12)) :=
by sorry

-- Define the translation function
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x + h)

-- Theorem stating that cos(2x + π/3) is a left translation of sin(2x)
theorem cos_2x_plus_pi_3_is_left_translation_of_sin_2x :
  ∃ h : ℝ, (∀ x : ℝ, cos (2 * x + π / 3) = (translate (λ x ↦ sin (2 * x)) h) x) ∧ h = 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_pi_3_graph_translation_cos_2x_plus_pi_3_is_left_translation_of_sin_2x_l1318_131827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_48_minutes_l1318_131878

-- Define the pool capacity in cubic meters
def pool_capacity : ℚ := 12000

-- Define the time for the first valve to fill the pool in hours
def first_valve_time : ℚ := 2

-- Define the additional water emitted by the second valve per minute
def second_valve_additional : ℚ := 50

-- Define the function to calculate the time to fill the pool with both valves
noncomputable def time_to_fill (capacity : ℚ) (first_valve_time : ℚ) (second_valve_additional : ℚ) : ℚ :=
  let first_valve_rate := capacity / (first_valve_time * 60)
  let second_valve_rate := first_valve_rate + second_valve_additional
  let combined_rate := first_valve_rate + second_valve_rate
  capacity / combined_rate

-- Theorem statement
theorem fill_time_is_48_minutes :
  time_to_fill pool_capacity first_valve_time second_valve_additional = 48 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_48_minutes_l1318_131878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heap_division_impossibility_l1318_131828

theorem heap_division_impossibility (heaps : List Nat) : 
  heaps.length = 31 → 
  heaps.sum = 660 → 
  (∀ x y, x ∈ heaps → y ∈ heaps → y < 2 * x) → 
  False :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heap_division_impossibility_l1318_131828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_l1318_131830

/-- A function f has a clever value point if there exists an x₀ such that f(x₀) = f'(x₀) -/
def has_clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = deriv f x₀

/-- The square function -/
def f1 : ℝ → ℝ := λ x ↦ x^2

/-- The exponential decay function -/
noncomputable def f2 : ℝ → ℝ := λ x ↦ Real.exp (-x)

/-- The natural logarithm function -/
noncomputable def f3 : ℝ → ℝ := λ x ↦ Real.log x

/-- The tangent function -/
noncomputable def f4 : ℝ → ℝ := λ x ↦ Real.tan x

/-- The function x + 1/x -/
noncomputable def f5 : ℝ → ℝ := λ x ↦ x + 1/x

theorem clever_value_points :
  has_clever_value_point f1 ∧
  ¬has_clever_value_point f2 ∧
  has_clever_value_point f3 ∧
  ¬has_clever_value_point f4 ∧
  has_clever_value_point f5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_l1318_131830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l1318_131880

/-- Predicate to define the centroid of a triangle -/
def is_centroid (M A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  M = (1/3) • (A + B + C)

/-- Given a triangle ABC with centroid M and an arbitrary point O in space,
    prove that OM² = (1/3)(OA² + OB² + OC²) - (1/9)(AB² + BC² + AC²) -/
theorem centroid_distance_relation (A B C M O : EuclideanSpace ℝ (Fin 3)) :
  is_centroid M A B C →
  ‖M - O‖^2 = (1/3) * (‖A - O‖^2 + ‖B - O‖^2 + ‖C - O‖^2) - (1/9) * (‖A - B‖^2 + ‖B - C‖^2 + ‖A - C‖^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l1318_131880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1318_131854

theorem sin_cos_identity (θ : Real) (a b : Real) 
  (h1 : Real.sin θ + Real.cos θ = a) 
  (h2 : Real.sin θ - Real.cos θ = b) : 
  a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1318_131854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l1318_131859

/-- The ratio of the total area of three congruent equilateral triangles
    to the area of a larger equilateral triangle formed from their perimeter -/
theorem equilateral_triangle_area_ratio (s : ℝ) (h : s > 0) :
  (3 * (Real.sqrt 3 / 4 * s^2)) / (Real.sqrt 3 / 4 * (3*s)^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_ratio_l1318_131859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1318_131840

-- Define the circle
def my_circle (x y : ℝ) (a : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + a = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem line_equation_proof (a : ℝ) (h1 : a < 3) :
  ∀ (A B : ℝ × ℝ),
    (∃ (x y : ℝ), my_circle x y a ∧ line_l x y) →
    (A.1 + B.1) / 2 = midpoint_AB.1 ∧ (A.2 + B.2) / 2 = midpoint_AB.2 →
    ∀ (x y : ℝ), (x, y) ∈ Set.Icc A B → line_l x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1318_131840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1318_131855

theorem triangle_area (A B C : ℝ) : 
  let angle_A := A * π / 180
  let AB := 2
  let BC := 1
  let AC := Real.sqrt (AB^2 - BC^2)
  let area := (1/2) * BC * AC
  A = 30 → area = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1318_131855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_digit_sum_is_370_l1318_131836

/-- The sequence of numbers from 1 to 55 written sequentially -/
def numberSequence : List Nat := List.range 55

/-- Calculates the sum of all digits in a given number -/
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

/-- Calculates the sum of all digits in the sequence -/
def sequenceDigitSum : Nat :=
  numberSequence.foldl (fun acc n => acc + sumDigits n) 0

/-- Theorem stating that the sum of all digits in the sequence is 370 -/
theorem sequence_digit_sum_is_370 : sequenceDigitSum = 370 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_digit_sum_is_370_l1318_131836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_with_properties_l1318_131858

/-- Represents a quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The axis of symmetry of a quadratic polynomial -/
noncomputable def axisOfSymmetry (q : QuadraticPolynomial) : ℝ := -q.b / (2 * q.a)

/-- The derivative of a quadratic polynomial at a point -/
def derivative (q : QuadraticPolynomial) (x : ℝ) : ℝ := 2 * q.a * x + q.b

/-- Two quadratic polynomials with specific properties -/
theorem quadratic_polynomials_with_properties :
  ∃ (f g : QuadraticPolynomial) (x₁ x₂ : ℝ),
    -- The polynomials intersect at two points
    (f.a * x₁^2 + f.b * x₁ + f.c = g.a * x₁^2 + g.b * x₁ + g.c) ∧
    (f.a * x₂^2 + f.b * x₂ + f.c = g.a * x₂^2 + g.b * x₂ + g.c) ∧
    (x₁ ≠ x₂) ∧
    -- The tangents at the intersection points are perpendicular
    (derivative f x₁ * derivative g x₁ = -1) ∧
    (derivative f x₂ * derivative g x₂ = -1) ∧
    -- The axes of symmetry do not coincide
    (axisOfSymmetry f ≠ axisOfSymmetry g) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_with_properties_l1318_131858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_final_distance_l1318_131832

noncomputable def hiker_distance (north south west east : ℝ) : ℝ :=
  Real.sqrt ((north - south)^2 + (west - east)^2)

theorem hiker_final_distance :
  hiker_distance 15 3 20 8 = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_final_distance_l1318_131832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_X_percentage_in_A_l1318_131866

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := sorry

/-- The weight of solution A in grams -/
def weight_A : ℝ := 400

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 0.018

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.0158

theorem liquid_X_percentage_in_A :
  percentage_X_in_A * weight_A + percentage_X_in_B * weight_B = 
  percentage_X_in_mixture * (weight_A + weight_B) →
  percentage_X_in_A = 0.01195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_X_percentage_in_A_l1318_131866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_partition_theorem_l1318_131844

-- Define a type for positive integers
def PositiveInt := {n : ℕ // n > 0}

-- Define a function to get the positive divisors of a number
def positiveDivisors (n : PositiveInt) : Set PositiveInt :=
  {d | d.val ∣ n.val ∧ d.val > 0}

-- Define a type for partitions of positive divisors into pairs
structure DivisorPartition (n : PositiveInt) where
  pairs : List (PositiveInt × PositiveInt)
  all_divisors : ∀ d, d ∈ positiveDivisors n → ∃ p ∈ pairs, d = p.1 ∨ d = p.2
  no_overlap : ∀ p q, p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2

-- Define a predicate for prime numbers
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Main theorem
theorem divisor_partition_theorem (n : PositiveInt) 
  (partition : DivisorPartition n)
  (sum_is_prime : ∀ p, p ∈ partition.pairs → isPrime (p.1.val + p.2.val)) :
  (∀ p q, p ∈ partition.pairs → q ∈ partition.pairs → p ≠ q → p.1.val + p.2.val ≠ q.1.val + q.2.val) ∧
  (∀ p, p ∈ partition.pairs → ¬(p.1.val + p.2.val ∣ n.val)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_partition_theorem_l1318_131844
