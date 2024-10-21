import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_time_and_percentage_l737_73742

def program_duration : ℚ := 30

def commercial_percentages : List ℚ := [1/5, 1/4, 0, 7/20, 2/5, 9/20, 3/20, 0]

def total_programs : ℕ := 8

def required_percentage : ℚ := 7/25

theorem commercial_time_and_percentage 
  (h1 : ∀ p ∈ commercial_percentages, 0 ≤ p ∧ p ≤ 1) 
  (h2 : commercial_percentages.length = total_programs) :
  let total_time := (total_programs : ℚ) * program_duration
  let commercial_time := (commercial_percentages.map (· * program_duration)).sum
  let overall_percentage := commercial_time / total_time
  commercial_time = 54 ∧ overall_percentage < required_percentage := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commercial_time_and_percentage_l737_73742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₃_equals_nine_l737_73719

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the points A, B, and E
noncomputable def A : ℝ × ℝ := (1, f 1)
noncomputable def B : ℝ × ℝ := (16, f 16)

-- Define the y-coordinate of point C
noncomputable def yC : ℝ := (1/3) * (f 1) + (2/3) * (f 16)

-- Define x₃ as the x-coordinate of point E
noncomputable def x₃ : ℝ := yC ^ 2

-- Theorem statement
theorem x₃_equals_nine : x₃ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₃_equals_nine_l737_73719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l737_73766

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1/2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

/-- Given an ellipse satisfying the problem conditions, proves its standard equation and the area of triangle BCD -/
theorem ellipse_properties (E : Ellipse) (F B C : Point) :
  F.x = -1 ∧ F.y = 0 ∧  -- Left focus
  B.x = 0 ∧ B.y = E.b ∧  -- Top vertex
  C.x = 0 ∧ C.y = -E.b ∧  -- Bottom vertex
  (∃ (M : Point), M.x = -E.a/2 ∧ M.y = -E.b/2 ∧  -- Midpoint of AC
    B.y - F.y = ((B.y - F.y)/(B.x - F.x)) * (M.x - F.x) + F.y) →  -- BF passes through M
  (∀ (x y : ℝ), x^2/9 + y^2/8 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧  -- Standard equation
  ((B.y - F.y)/(B.x - F.x) = 1 →  -- Slope of BF is 1
    ∃ (D : Point), D.x ≠ B.x ∧ D.y ≠ B.y ∧
      D.x^2/(E.a^2) + D.y^2/(E.b^2) = 1 ∧  -- D is on the ellipse
      D.y - B.y = (D.x - B.x) ∧  -- D is on line BF
      area_triangle B C D = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l737_73766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l737_73739

def is_valid (n : ℕ) : Prop :=
  Nat.Prime (n^n + 1) ∧ (n^n + 1 : ℕ) < 10^18

theorem valid_n_characterization :
  {n : ℕ | n > 0 ∧ is_valid n} = {1, 2, 4} := by
  sorry

#check valid_n_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l737_73739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_roaming_area_is_19pi_l737_73728

/-- The area that Abby the alpaca can roam when tethered to the corner of a rectangular barn --/
noncomputable def abby_roaming_area (barn_width : ℝ) (barn_length : ℝ) (leash_length : ℝ) : ℝ :=
  let large_segment := (3 / 4) * Real.pi * leash_length^2
  let small_segment := (1 / 4) * Real.pi * (leash_length - barn_width)^2
  large_segment + small_segment

/-- Theorem stating that Abby's roaming area is 19π square meters --/
theorem abby_roaming_area_is_19pi :
  abby_roaming_area 4 6 5 = 19 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_roaming_area_is_19pi_l737_73728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_hula_hoops_l737_73713

def hula_hoops : Nat → Nat
  | 0 => 18  -- We define day 0 as 18 to match day 1 in the problem
  | n + 1 => 2 * hula_hoops n

theorem fourth_day_hula_hoops : hula_hoops 3 = 144 := by
  -- Unfold the definition for days 1, 2, and 3
  calc hula_hoops 3
    = 2 * hula_hoops 2 := by rfl
    _ = 2 * (2 * hula_hoops 1) := by rfl
    _ = 2 * (2 * (2 * hula_hoops 0)) := by rfl
    _ = 2 * (2 * (2 * 18)) := by rfl
    _ = 144 := by norm_num

#eval hula_hoops 3  -- This will output 144

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_hula_hoops_l737_73713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sin_cos_product_negative_l737_73735

/-- An angle is in the second quadrant if its cosine is negative and its sine is positive -/
def is_second_quadrant (θ : Real) : Prop := Real.cos θ < 0 ∧ Real.sin θ > 0

/-- The main theorem: for any angle in the second quadrant, 
    the product sin(cos θ) * cos(sin 2θ) is negative -/
theorem second_quadrant_sin_cos_product_negative (θ : Real) 
  (h : is_second_quadrant θ) : Real.sin (Real.cos θ) * Real.cos (Real.sin (2 * θ)) < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_sin_cos_product_negative_l737_73735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l737_73793

/-- A system of equations with parameter a -/
def SystemOfEquations (a x y : ℝ) : Prop :=
  (x - a)^2 = 9 * (y - x + a - 2) ∧ (x/2 : ℝ) ^ (y/2) = 2

/-- The theorem stating the unique solution to the system -/
theorem unique_solution :
  ∃! a : ℝ, ∃ x y : ℝ, SystemOfEquations a x y ∧ a = 11 ∧ x = 20 ∧ y = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l737_73793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option2_more_profitable_l737_73751

/-- Represents the profit function for the company's investment -/
noncomputable def profit (n : ℕ) : ℝ := -10 * n^2 + 100 * n - 160

/-- Represents the average profit function for the company's investment -/
noncomputable def avgProfit (n : ℕ) : ℝ := profit n / n

/-- The number of years that maximizes the total profit -/
def maxTotalProfitYear : ℕ := 5

/-- The number of years that maximizes the average profit -/
def maxAvgProfitYear : ℕ := 4

/-- The selling price for Option 1 -/
def sellingPrice1 : ℝ := 200

/-- The selling price for Option 2 -/
def sellingPrice2 : ℝ := 300

/-- The initial investment -/
def initialInvestment : ℝ := 160

theorem option2_more_profitable :
  profit maxAvgProfitYear + sellingPrice2 - initialInvestment >
  profit maxTotalProfitYear + sellingPrice1 - initialInvestment := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option2_more_profitable_l737_73751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l737_73760

theorem det_product {n : Type*} [Fintype n] [DecidableEq n] (A B C : Matrix n n ℝ) 
  (hA : Matrix.det A = 3)
  (hB : Matrix.det B = -7)
  (hC : Matrix.det C = 4) :
  Matrix.det (A * B * C) = -84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_product_l737_73760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_exponential_equation_l737_73705

theorem solution_of_exponential_equation :
  ∃! x : ℝ, (3 : ℝ)^(-x) = 2 + (3 : ℝ)^(x+1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_exponential_equation_l737_73705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_profit_and_max_avg_profit_l737_73706

/-- Represents the profit function for Xiao Li's contracted land -/
noncomputable def profit_function (x : ℝ) : ℝ := -x^2 + 20*x - 36

/-- Represents the annual average profit function for Xiao Li's contracted land -/
noncomputable def avg_profit_function (x : ℝ) : ℝ := -x + 20 - 36/x

/-- The fixed cost investment in thousand yuan -/
def fixed_cost : ℝ := 360

/-- The annual total income in thousand yuan -/
def annual_income : ℝ := 220

theorem xiao_li_profit_and_max_avg_profit :
  (∃ x : ℝ, x > 2 ∧ x < 3 ∧ profit_function x > 0) ∧
  (∀ x : ℝ, x > 0 → avg_profit_function x ≤ 8) ∧
  (avg_profit_function 6 = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiao_li_profit_and_max_avg_profit_l737_73706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_numbers_on_circle_l737_73771

theorem complex_numbers_on_circle (a₁ a₂ a₃ a₄ a₅ : ℂ) (s : ℝ) :
  (a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0) →
  (a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅)) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = s) →
  (Complex.abs s ≤ 2) →
  ∃ q : ℂ, Complex.abs q = 1 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_numbers_on_circle_l737_73771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_of_S_and_consecutive_pairs_l737_73732

/-- The set S of positive integers x for which there exist positive integers y and m
    such that y^2 - 2^m = x^2 -/
def S : Set ℕ+ := {x | ∃ y m : ℕ+, (y : ℤ)^2 - (2 : ℤ)^(m : ℕ) = (x : ℤ)^2}

theorem elements_of_S_and_consecutive_pairs :
  (1 : ℕ+) ∈ S ∧ (2 : ℕ+) ∈ S ∧ (3 : ℕ+) ∈ S ∧
  (∀ x : ℕ+, x ∈ S ∧ (x + 1) ∈ S ↔ x = 1 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_of_S_and_consecutive_pairs_l737_73732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_a_approx_l737_73743

/-- The weight of one liter of vegetable ghee of brand 'b' in grams -/
noncomputable def weight_b : ℝ := 850

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
noncomputable def mixture_ratio : ℝ := 3 / 2

/-- The total volume of the mixture in liters -/
noncomputable def total_volume : ℝ := 3

/-- The total weight of the mixture in grams -/
noncomputable def total_weight : ℝ := 2460

/-- The weight of one liter of vegetable ghee of brand 'a' in grams -/
noncomputable def weight_a : ℝ := (total_weight - (weight_b * total_volume / (1 + mixture_ratio))) * 
                    (1 + mixture_ratio) / (mixture_ratio * total_volume)

theorem weight_a_approx : 
  988.8 < weight_a ∧ weight_a < 989 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_a_approx_l737_73743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_theorem_l737_73717

/-- Represents the odometer reading as a function of digits a, b, and c -/
def odometer_reading (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the conditions of the problem -/
structure TripData where
  a : ℕ
  b : ℕ
  c : ℕ
  hours : ℕ
  speed : ℕ
  start_reading : ℕ
  end_reading : ℕ
  h1 : a ≥ 1
  h2 : a + b + c ≤ 7
  h3 : speed = 55
  h4 : start_reading = odometer_reading a b c
  h5 : end_reading = odometer_reading c b a
  h6 : end_reading - start_reading = speed * hours

/-- The main theorem to be proved -/
theorem trip_theorem (data : TripData) : data.a^2 + data.b^2 + data.c^2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_theorem_l737_73717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l737_73773

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 1 ∨ (1 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l737_73773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_winning_strategy_one_winning_strategy_l737_73740

/-- Represents a rectangular container with base area and height --/
structure Container where
  base_area : ℝ
  height : ℝ

/-- The game setup with four containers --/
def GameSetup (a b : ℝ) : (Container × Container × Container × Container) :=
  ({ base_area := a^2, height := a },   -- Container A
   { base_area := a^2, height := b },   -- Container B
   { base_area := b^2, height := a },   -- Container C
   { base_area := b^2, height := b })   -- Container D

/-- The volume of a container --/
def volume (c : Container) : ℝ := c.base_area * c.height

/-- The total volume of two containers --/
def total_volume (c1 c2 : Container) : ℝ := volume c1 + volume c2

/-- Predicate to check if a strategy is winning --/
def IsWinningStrategy (strategy : String) (a b : ℝ) : Prop :=
  let (A, B, C, D) := GameSetup a b
  match strategy with
  | "Choose A and D" => total_volume A D > total_volume B C
  | _ => False

/-- Theorem stating that choosing containers A and D is the only winning strategy --/
theorem only_winning_strategy (a b : ℝ) (h : a ≠ b) :
  let (A, B, C, D) := GameSetup a b
  (total_volume A D > total_volume B C) ∧
  (total_volume A B ≠ total_volume C D) ∧
  (total_volume A C ≠ total_volume B D) := by
  sorry

/-- Corollary: There is exactly one winning strategy for player A --/
theorem one_winning_strategy (a b : ℝ) (h : a ≠ b) :
  ∃! strategy, strategy = "Choose A and D" ∧ IsWinningStrategy strategy a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_winning_strategy_one_winning_strategy_l737_73740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_bisector_l737_73795

/-- A point P(x, y) lies on the angle bisector in the first quadrant if and only if x = y -/
def IsOnAngleBisector (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 = p.2

/-- The theorem states that if a point P(1-2a, a-2) lies on the angle bisector in the first quadrant,
    then a = 1 -/
theorem point_on_angle_bisector (a : ℝ) : 
  IsOnAngleBisector (1 - 2*a, a - 2) → a = 1 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_bisector_l737_73795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l737_73794

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / crossing_time
  3.6 * speed_mps

/-- Theorem stating the speed of the train given the problem conditions -/
theorem train_speed_problem : 
  |train_speed 100 145 13.568145317605362 - 64.9944| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l737_73794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l737_73763

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) - 2 * (Real.cos x) ^ 2

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) where
  -- No additional conditions needed

-- Define the theorem
theorem triangle_side_length 
  (A B C D : ℝ × ℝ) 
  (triangle : Triangle A B C) 
  (h1 : f ((Real.arccos (1/4)) / 2 - Real.pi / 6) = -5/4)
  (h2 : D.1 - C.1 = 2 * (A.1 - D.1) ∧ D.2 - C.2 = 2 * (A.2 - D.2))
  (h3 : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = Real.sqrt 10)
  (h4 : (B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
        (D.1 - A.1) / Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) +
        (B.2 - A.2) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * 
        (D.2 - A.2) / Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = Real.sqrt 10 / 4) :
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l737_73763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_element_is_three_l737_73752

def create_modified_list : List ℕ :=
  (List.range 10).map (· + 11) ++ List.range 10 ++ [10]

theorem thirteenth_element_is_three : 
  (create_modified_list.get? 12).isSome ∧ 
  (create_modified_list.get? 12).get! = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_element_is_three_l737_73752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_equal_surface_area_l737_73736

/-- Given a cube with side length 6.5 cm and a sphere with the same surface area,
    prove that the radius of the sphere rounded to the nearest whole number is 4 cm. -/
theorem cube_sphere_equal_surface_area :
  let cube_side : ℝ := 6.5
  let cube_surface_area : ℝ := 6 * cube_side ^ 2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_radius : ℝ := Real.sqrt (sphere_surface_area / (4 * Real.pi))
  (Int.toNat (round sphere_radius)) = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_equal_surface_area_l737_73736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_current_in_circuit_l737_73734

-- Define complex numbers
def V : ℂ := 2 - 2*Complex.I
def Z1 : ℂ := 3 + 4*Complex.I
def Z2 : ℂ := 1 - Complex.I

-- Define total impedance
def Z : ℂ := Z1 + Z2

-- Define total current
noncomputable def I : ℂ := V / Z

-- Theorem statement
theorem total_current_in_circuit :
  I = 14/25 - 14/25*Complex.I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_current_in_circuit_l737_73734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l737_73785

/-- Given a hyperbola with the following properties:
    - P is a point on the hyperbola (x²/a²) - (y²/b²) = 1
    - a > 0, b > 0
    - F₁ and F₂ are the foci of the hyperbola
    - PF₁ ⊥ PF₂
    - Area of triangle PF₁F₂ is 1
    - a + b = 3
    Prove that the eccentricity of the hyperbola is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) = 0 →
  (1/2) * abs ((F₂.1 - F₁.1) * (P.2 - F₁.2) - (F₂.2 - F₁.2) * (P.1 - F₁.1)) = 1 →
  a + b = 3 →
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 5 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l737_73785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l737_73783

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) 
  (donation_percent : ℚ) (savings_percent : ℚ) :
  total_cookies = 5825 →
  num_people = 23 →
  donation_percent = 12 / 100 →
  savings_percent = 5 / 100 →
  ((total_cookies : ℚ) - 
    (donation_percent * total_cookies) - 
    (savings_percent * total_cookies)).floor / num_people = 210 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_distribution_l737_73783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_integer_l737_73787

/-- Given integers a and b with a > b > 0, and θ ∈ (0, π/2) such that
    sin θ = (2ab) / (a^2 + b^2), prove that A_n is an integer for all natural numbers n,
    where A_n = (a^2 + b^2)^n * sin(n * θ) -/
theorem A_n_is_integer (a b : ℤ) (θ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : θ ∈ Set.Ioo 0 (π / 2)) (h4 : Real.sin θ = (2 * a * b : ℝ) / (a^2 + b^2 : ℝ)) :
  ∀ n : ℕ, ∃ k : ℤ, (a^2 + b^2 : ℝ)^n * Real.sin (n * θ) = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_integer_l737_73787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_implies_m_values_l737_73720

-- Define the circle
def myCircle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line
def myLine (x y : ℝ) (m : ℝ) : Prop := x - m * y + 1 = 0

-- Define the intersection points
def intersection_points (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    myCircle x1 y1 ∧ myCircle x2 y2 ∧
    myLine x1 y1 m ∧ myLine x2 y2 m ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the triangle area condition
def triangle_area_condition (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    myCircle x1 y1 ∧ myCircle x2 y2 ∧
    myLine x1 y1 m ∧ myLine x2 y2 m ∧
    (1/2 * |x1 * y2 - x2 * y1 + x2 - x1| = 8/5)

-- Theorem statement
theorem intersection_and_area_implies_m_values (m : ℝ) :
  intersection_points m ∧ triangle_area_condition m →
  m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_implies_m_values_l737_73720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l737_73755

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the conditions
def triangle_conditions (A B C a b c : ℝ) : Prop :=
  a > b ∧ a = 5 ∧ c = 6 ∧ Real.sin B = 3/5

-- Theorem for part 1
theorem part_one (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) : 
  b = Real.sqrt 13 ∧ Real.sin A = (3 * Real.sqrt 13) / 13 := by
  sorry

-- Theorem for part 2
theorem part_two (A B C a b c : ℝ) (h : triangle_conditions A B C a b c) : 
  Real.sin (2*A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l737_73755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_squared_l737_73790

/-- Predicate to check if a, b, c form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if m is a median of the triangle with sides a, b, c -/
def is_median (a b c m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = (b^2 + c^2) / 4 ∧ m^2 = a^2 / 4 + x^2 - a * x / 2

/-- Triangle inequality for squared sides and medians -/
theorem triangle_inequality_squared (a b c m_a m_b : ℝ) 
  (h_triangle : is_triangle a b c) 
  (h_median_a : is_median a b c m_a)
  (h_median_b : is_median a b c m_b) :
  (a^2 + b^2 ≥ c^2 / 2) ∧ (m_a^2 + m_b^2 ≥ 9 * c^2 / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_squared_l737_73790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_theorem_l737_73762

theorem constant_function_theorem (f : ℝ → ℝ) (k : ℝ) :
  (k ≠ 0) →
  (∀ x y : ℝ, x > 0 → y > 0 → k * (x * f y - y * f x) = f (x / y)) →
  f 100 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_theorem_l737_73762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_last_two_digits_l737_73799

theorem sum_of_factorials_last_two_digits : 
  ((List.range 14).map (fun n => Nat.factorial (n + 1))).sum % 100 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factorials_last_two_digits_l737_73799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l737_73778

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.sin x * (Real.cos x + Real.sin x) - Real.sqrt 2 / 2

theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ f x) ∧
  f x = -Real.sqrt 2 / 2 := by
  sorry

#check min_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l737_73778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_4x_plus_y_equals_sqrt_2_l737_73709

theorem sqrt_4x_plus_y_equals_sqrt_2 (x y : ℝ) : 
  y = 1 + Real.sqrt (4 * x - 1) + Real.sqrt (1 - 4 * x) →
  Real.sqrt (4 * x + y) = Real.sqrt 2 ∨ Real.sqrt (4 * x + y) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_4x_plus_y_equals_sqrt_2_l737_73709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_specific_l737_73769

/-- The area a goat can graze when tied to the corner of a rectangular barn -/
noncomputable def goat_grazing_area (barn_length barn_width leash_length : ℝ) : ℝ :=
  (3/4) * Real.pi * leash_length^2 + (1/4) * Real.pi * (leash_length - barn_length)^2

/-- Theorem: The area a goat can graze when tied to the corner of a 4m by 3m barn 
    with a 4m leash, moving only around the outside of the barn, is 12.25π square meters -/
theorem goat_grazing_area_specific : goat_grazing_area 4 3 4 = 12.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goat_grazing_area_specific_l737_73769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l737_73772

-- Define the square side length and circle radius
def squareSide : ℝ := 8
def outerCircleRadius : ℝ := 3

-- Define the shaded area function
noncomputable def shadedArea (s : ℝ) (r : ℝ) : ℝ :=
  s^2 - (Real.pi * r^2) - (Real.pi * (s/2 - r)^2)

-- State the theorem
theorem shaded_area_calculation :
  shadedArea squareSide outerCircleRadius = 64 - 10 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l737_73772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l737_73750

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The point A -/
def A : ℝ × ℝ := (-3, 0)

/-- The point B -/
def B : ℝ × ℝ := (2, 5)

/-- The point on the y-axis -/
def P : ℝ × ℝ := (0, 2)

theorem equidistant_point : 
  distance P.1 P.2 A.1 A.2 = distance P.1 P.2 B.1 B.2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l737_73750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_51_l737_73777

def g (x : ℤ) : ℤ := x^2 + x + 2023

theorem gcd_g_50_51 : Int.gcd (g 50) (g 51) = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_51_l737_73777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l737_73761

/-- Parabola type representing y² = 2px --/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola --/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Focus of a parabola --/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- Line passing through a point at a given angle --/
structure Line where
  point : ℝ × ℝ
  angle : ℝ

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement --/
theorem parabola_intersection_ratio (c : Parabola) (l : Line) 
  (h_focus : l.point = focus c)
  (h_angle : l.angle = π/3) -- 60 degrees in radians
  (A B : ParabolaPoint c)
  (h_A_on_l : A.x * Real.sin l.angle = A.y * Real.cos l.angle)
  (h_B_on_l : B.x * Real.sin l.angle = B.y * Real.cos l.angle)
  (h_A_first_quadrant : A.x > 0 ∧ A.y > 0)
  (h_B_fourth_quadrant : B.x > 0 ∧ B.y < 0) :
  distance (A.x, A.y) (focus c) / distance (B.x, B.y) (focus c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l737_73761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_intersection_l737_73747

-- Define the basic structures
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the non-intersecting property
def non_intersecting (c1 c2 : Circle) : Prop := 
  dist c1.center c2.center > c1.radius + c2.radius

-- Define tangent points
def is_tangent_point (p : EuclideanSpace ℝ (Fin 2)) (c : Circle) : Prop :=
  dist p c.center = c.radius

-- Define collinearity
def collinear (p q r : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, r = p + t • (q - p)

-- Define a line
def line (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {r | collinear p q r}

-- Define the theorem
theorem tangent_chord_intersection 
  (c1 c2 : Circle) 
  (A1 A2 B1 B2 : EuclideanSpace ℝ (Fin 2)) :
  non_intersecting c1 c2 →
  is_tangent_point A1 c1 →
  is_tangent_point A2 c2 →
  is_tangent_point B1 c1 →
  is_tangent_point B2 c2 →
  collinear A1 A2 B1 →
  collinear A1 A2 B2 →
  ∃ P, P ∈ line A1 B1 ∧ P ∈ line A2 B2 ∧ P ∈ line c1.center c2.center :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_chord_intersection_l737_73747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_sum_of_coordinates_l737_73708

-- Define the functions h and j
noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

-- State the given conditions
axiom intersect_3 : h 3 = j 3 ∧ h 3 = 3
axiom intersect_5 : h 5 = j 5 ∧ h 5 = 10
axiom intersect_7 : h 7 = j 7 ∧ h 7 = 21
axiom intersect_9 : h 9 = j 9 ∧ h 9 = 21

-- Theorem to prove
theorem intersection_of_transformed_functions :
  h (3 * 3) = 3 * (j 3) ∧ h (3 * 3) = 21 := by
  sorry

-- Theorem to prove the sum of coordinates
theorem sum_of_coordinates : 3 + 21 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_transformed_functions_sum_of_coordinates_l737_73708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l737_73710

theorem sin_transformation (x : ℝ) :
  Real.sin (2 * x) = Real.sin ((2 * (x + π/8)) + π/4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transformation_l737_73710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l737_73712

/-- The time taken for two trains to cross each other -/
noncomputable def crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed2 - speed1) * (1000 / 3600))

/-- Theorem stating the crossing time for the given problem -/
theorem train_crossing_time :
  let length1 : ℝ := 150  -- length of Train 1 in meters
  let length2 : ℝ := 200  -- length of Train 2 in meters
  let speed1 : ℝ := 25    -- speed of Train 1 in kmph
  let speed2 : ℝ := 35    -- speed of Train 2 in kmph
  abs (crossing_time length1 length2 speed1 speed2 - 125.9) < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval crossing_time 150 200 25 35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l737_73712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_pyramid_volume_positive_l737_73707

/-- The volume of a pyramid with an isosceles triangle base -/
theorem pyramid_volume (a α β : ℝ) (ha : a > 0) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π/2) :
  ∃ V : ℝ, V = (1/6) * a^3 * Real.sin (α/2) * Real.tan β :=
by
  -- We define V as the volume of the pyramid
  let V := (1/6) * a^3 * Real.sin (α/2) * Real.tan β
  
  -- We assert that this V satisfies the equation
  have h : V = (1/6) * a^3 * Real.sin (α/2) * Real.tan β := by rfl
  
  -- We conclude that there exists such a V
  exact ⟨V, h⟩

/-- The volume formula is positive given positive inputs -/
theorem pyramid_volume_positive (a α β : ℝ) (ha : a > 0) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π/2) :
  (1/6) * a^3 * Real.sin (α/2) * Real.tan β > 0 :=
by
  sorry  -- The detailed proof is omitted for brevity

-- Additional theorems or lemmas could be added here to further prove properties of the formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_pyramid_volume_positive_l737_73707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_walking_home_fraction_l737_73701

theorem students_walking_home_fraction :
  let bus_fraction : ℚ := 1/3
  let car_fraction : ℚ := 1/5
  let skateboard_fraction : ℚ := 1/8
  let shared_ride_fraction : ℚ := 1/10
  let total_other_transport : ℚ := bus_fraction + car_fraction + skateboard_fraction + shared_ride_fraction
  let walking_fraction : ℚ := 1 - total_other_transport
  walking_fraction = 29/120 := by
  -- Proof steps would go here
  sorry

#check students_walking_home_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_walking_home_fraction_l737_73701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_trapezoid_properties_l737_73729

/-- Represents a right trapezoid ABCD -/
structure RightTrapezoid where
  AB : ℝ
  BC : ℝ
  AD : ℝ
  right_angle : AB = 1 -- AB ⊥ BC implies AB = 1 given BC = 2
  parallel : BC = AD + AB -- BC || AD
  dimensions : BC = 2 ∧ AB = 1 ∧ AD = 1

/-- The surface area of the solid formed by rotating the trapezoid around AB -/
noncomputable def surface_area (t : RightTrapezoid) : ℝ := (5 + 3 * Real.sqrt 2) * Real.pi

/-- The volume of the solid formed by rotating the trapezoid around AB -/
noncomputable def volume (t : RightTrapezoid) : ℝ := 7 * Real.pi / 3

/-- Theorem stating the surface area and volume of the rotated solid -/
theorem rotated_trapezoid_properties (t : RightTrapezoid) :
  surface_area t = (5 + 3 * Real.sqrt 2) * Real.pi ∧
  volume t = 7 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_trapezoid_properties_l737_73729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_f_always_positive_l737_73718

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a / x - 2)

-- Theorem for part (1)
theorem min_value_of_f (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  ∀ x ≥ 2, f a x ≥ f a 2 := by
  sorry

-- Theorem for part (2)
theorem f_always_positive (a : ℝ) :
  (∀ x ≥ 2, f a x > 0) ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_f_always_positive_l737_73718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_consumption_rate_l737_73741

/-- Represents the gasoline consumption problem --/
structure GasolineConsumption where
  initial_gas : ℚ
  final_gas : ℚ
  supermarket_distance : ℚ
  farm_distance : ℚ
  partial_farm_distance : ℚ

/-- Calculates the total distance driven --/
def total_distance (gc : GasolineConsumption) : ℚ :=
  2 * gc.supermarket_distance + 2 * gc.partial_farm_distance + gc.farm_distance

/-- Calculates the total gasoline used --/
def gas_used (gc : GasolineConsumption) : ℚ :=
  gc.initial_gas - gc.final_gas

/-- Calculates the rate of gasoline consumption --/
def consumption_rate (gc : GasolineConsumption) : ℚ :=
  total_distance gc / gas_used gc

/-- Theorem stating the rate of gasoline consumption --/
theorem gasoline_consumption_rate (gc : GasolineConsumption) 
  (h1 : gc.initial_gas = 12)
  (h2 : gc.final_gas = 2)
  (h3 : gc.supermarket_distance = 5)
  (h4 : gc.farm_distance = 6)
  (h5 : gc.partial_farm_distance = 2) :
  consumption_rate gc = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_consumption_rate_l737_73741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l737_73779

-- Define the slopes of two lines
noncomputable def slope1 (a : ℝ) : ℝ := -1 / (2 * a)
noncomputable def slope2 (a : ℝ) : ℝ := -a / 4

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop := slope1 a = slope2 a

-- Theorem statement
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel_lines a ↔ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l737_73779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_min_root_product_l737_73796

theorem min_k_for_min_root_product (k : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0 ∧ x ≠ y) → 
  (∀ k' : ℝ, (∃ x' y' : ℝ, 2 * x'^2 + 5 * x' + k' = 0 ∧ 2 * y'^2 + 5 * y' + k' = 0 ∧ x' ≠ y') → 
    (∀ x y x' y' : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0 ∧ x ≠ y ∧
                      2 * x'^2 + 5 * x' + k' = 0 ∧ 2 * y'^2 + 5 * y' + k' = 0 ∧ x' ≠ y' →
                      x * y ≤ x' * y')) → 
  k = 25/8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_min_root_product_l737_73796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l737_73704

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x / (x + 1)

theorem problem_solution (a b : ℝ) :
  (∀ x, f a x < 0 ↔ a < x ∧ x < 1) →
  g a b = b + a/2 →
  a*b < 0 →
  (a = -4 ∧ ∀ c d, c*d < 0 → g c d = d + c/2 → 4*c + d ≤ 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l737_73704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_atomic_weight_l737_73748

/-- The atomic weight of an element -/
def atomic_weight (element : String) : ℝ := sorry

/-- The molecular weight of a compound -/
def molecular_weight (compound : String) : ℝ := sorry

/-- The number of atoms of an element in a compound -/
def atom_count (element : String) (compound : String) : ℕ := sorry

theorem barium_atomic_weight :
  atomic_weight "Cl" = 35.45 →
  molecular_weight "BaCl2" = 207 →
  atom_count "Ba" "BaCl2" = 1 →
  atom_count "Cl" "BaCl2" = 2 →
  atomic_weight "Ba" = 136.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barium_atomic_weight_l737_73748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l737_73798

/-- Triangle ABC with point D on BC, and inscribed circles in ADC and ADB -/
structure TriangleWithInscribedCircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  AB : ℝ
  BC : ℝ
  CA : ℝ
  BD_DC_ratio : ℝ

/-- The configuration satisfies the given conditions -/
def satisfies_conditions (t : TriangleWithInscribedCircles) : Prop :=
  t.AB = 14 ∧
  t.BC = 6 ∧
  t.CA = 9 ∧
  t.BD_DC_ratio = 1/9 ∧
  (t.D.1 - t.B.1) / (t.C.1 - t.D.1) = t.BD_DC_ratio ∧
  (t.D.2 - t.B.2) / (t.C.2 - t.D.2) = t.BD_DC_ratio

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem EF_length (t : TriangleWithInscribedCircles) 
  (h : satisfies_conditions t) : 
  distance t.E t.F = 4.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l737_73798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_of_y_l737_73702

/-- 
Given a total amount of 70 rupees divided among x, y, and z in the ratio of 100:45:30,
prove that the share of y is 18 rupees.
-/
theorem share_of_y (total : ℚ) (x y z : ℚ) : 
  total = 70 →
  x + y + z = total →
  100 * y = 45 * x ∧ 100 * z = 30 * x →
  y = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_of_y_l737_73702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l737_73722

noncomputable section

-- Define the expression
noncomputable def original_expression : ℝ := 
  (Real.sqrt 5 - 2)^(2 - Real.sqrt 6) / (Real.sqrt 5 + 2)^(2 + Real.sqrt 6)

-- Define the simplified expression
noncomputable def simplified_expression : ℝ := 
  1 / (9 - 4 * Real.sqrt 5)^(Real.sqrt 6)

-- Theorem statement
theorem expression_simplification :
  original_expression = simplified_expression :=
by sorry

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l737_73722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_and_star_operation_l737_73756

-- Define the new operation ※
noncomputable def star (x y : ℝ) : ℝ := Real.sqrt (x + y) / (x - y)

-- State the theorem
theorem quadratic_radical_and_star_operation :
  ∃ (a : ℝ), (2 * a - 2 = -a + 16) ∧
  (Real.sqrt a = Real.sqrt 6 ∨ Real.sqrt a = -Real.sqrt 6) ∧
  (star a (star a (-2)) = 10 / 23) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_and_star_operation_l737_73756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lateral_face_eq_l737_73744

/-- A regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  -- Height of the pyramid
  height : ℝ
  -- Angle between lateral edge and base plane
  lateral_angle : ℝ
  -- Assertion that the height is 6√6
  height_eq : height = 6 * Real.sqrt 6
  -- Assertion that the lateral angle is 45°
  angle_eq : lateral_angle = π / 4

/-- The distance from the center of the base to a lateral face -/
noncomputable def distance_to_lateral_face (p : RegularTriangularPyramid) : ℝ := 
  (6 * Real.sqrt 30) / 5

/-- Theorem stating the distance from the center of the base to a lateral face -/
theorem distance_to_lateral_face_eq (p : RegularTriangularPyramid) :
  distance_to_lateral_face p = (6 * Real.sqrt 30) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lateral_face_eq_l737_73744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_symmetric_points_on_parabola_l737_73724

/-- Two points are symmetric with respect to the line x + y = 0 -/
def symmetric (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.2 ∧ B.2 = -A.1

/-- A point lies on the parabola y = 3 - x^2 -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = 3 - p.1^2

/-- The distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem distance_of_symmetric_points_on_parabola (A B : ℝ × ℝ) :
  A ≠ B →
  on_parabola A →
  on_parabola B →
  symmetric A B →
  distance A B = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_symmetric_points_on_parabola_l737_73724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_mixture_density_l737_73749

/-- Given three liquids with densities in ratio 6:3:2, prove that the mixture density
    equals the arithmetic mean of the original densities iff 4x + 15y = 7,
    where x and y are mass ratios of the second and third liquids to the first, respectively. -/
theorem liquid_mixture_density
  (ρ₁ ρ₂ ρ₃ : ℝ)  -- Densities of the three liquids
  (h_ratio : ρ₁ / ρ₂ = 2 ∧ ρ₂ / ρ₃ = 3/2)  -- Density ratio condition
  (x y : ℝ)  -- Mass ratios
  (h_x : x ≤ 2/7)  -- Condition on x
  : (1 + x + y) / (1/6 + x/3 + y/2) = (ρ₁ + ρ₂ + ρ₃) / 3 ↔ 4*x + 15*y = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_mixture_density_l737_73749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l737_73721

noncomputable section

open Real

def f (ω a x : ℝ) : ℝ := sin (2 * ω * x + π / 3) + sqrt 3 / 2 + a

def g (f : ℝ → ℝ) (a x : ℝ) : ℝ := f x - a

theorem function_properties (ω a : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x > 0 → f ω a x ≤ f ω a (π / 6)) ∧ 
  (∀ x : ℝ, -π / 3 ≤ x ∧ x ≤ 5 * π / 6 → f ω a x ≥ sqrt 3) →
  ω = 1 / 2 ∧
  a = (sqrt 3 + 1) / 2 ∧
  (∀ x : ℝ, g (f (1 / 2) ((sqrt 3 + 1) / 2)) ((sqrt 3 + 1) / 2) x = sin (x + π / 3) + sqrt 3 / 2) ∧
  (∀ k : ℤ, ∀ x : ℝ, g (f (1 / 2) ((sqrt 3 + 1) / 2)) ((sqrt 3 + 1) / 2) (π / 6 + k * π - x) = 
    g (f (1 / 2) ((sqrt 3 + 1) / 2)) ((sqrt 3 + 1) / 2) (π / 6 + k * π + x)) ∧
  (∀ k : ℤ, ∀ x : ℝ, g (f (1 / 2) ((sqrt 3 + 1) / 2)) ((sqrt 3 + 1) / 2) (-π / 3 + k * π + x) + 
    g (f (1 / 2) ((sqrt 3 + 1) / 2)) ((sqrt 3 + 1) / 2) (-π / 3 + k * π - x) = sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l737_73721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_is_deductive_l737_73788

-- Define the universe of discourse
variable (Person : Type)

-- Define the predicate for making mistakes
variable (makesMistakes : Person → Prop)

-- Define Old Wang as a person
variable (oldWang : Person)

-- State the premises and conclusion
variable (everyone_makes_mistakes : ∀ (p : Person), makesMistakes p)

-- Define deductive reasoning
def isDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop :=
  (premise1 ∧ premise2) → conclusion

-- Theorem statement
theorem reasoning_is_deductive :
  isDeductiveReasoning
    (∀ (p : Person), makesMistakes p)
    (oldWang = oldWang)  -- This replaces (oldWang ∈ Person)
    (makesMistakes oldWang) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reasoning_is_deductive_l737_73788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_english_l737_73711

/-- Given a class with the following properties:
  * There are 40 students in total
  * 12 students are enrolled in both English and German
  * 22 students are enrolled in German
  * Every student is enrolled in at least one of English or German
  This theorem proves that 18 students are enrolled only in English. -/
theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) 
  (english : Finset ℕ) (german_set : Finset ℕ) (students : Finset ℕ)
  (h_total : students.card = 40)
  (h_both : (english ∩ german_set).card = 12)
  (h_german : german_set.card = 22)
  (h_all_enrolled : ∀ s ∈ students, s ∈ english ∨ s ∈ german_set)
  : (english \ german_set).card = 18 :=
by
  sorry

#check students_only_english

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_only_english_l737_73711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distances_impossibility_l737_73731

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in a 2D plane -/
structure Square where
  vertices : Fin 4 → Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- 
Theorem: It is impossible for a point P to have distances 1, 1, 2, and 3
to the vertices of a square in a 2D plane.
-/
theorem square_distances_impossibility (s : Square) (p : Point) :
  ¬ ∃ (perm : Fin 4 → Fin 4), 
    (distance p (s.vertices (perm 0)) = 1) ∧
    (distance p (s.vertices (perm 1)) = 1) ∧
    (distance p (s.vertices (perm 2)) = 2) ∧
    (distance p (s.vertices (perm 3)) = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_distances_impossibility_l737_73731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_preserved_different_invariants_transformation_impossible_l737_73723

/-- Represents a 2x2 matrix of integers modulo 26 -/
structure Matrix2x2 where
  a : Fin 26
  b : Fin 26
  c : Fin 26
  d : Fin 26

/-- Calculates the invariant k for a 2x2 matrix -/
def invariant (m : Matrix2x2) : Int :=
  (m.a.val + m.d.val) - (m.b.val + m.c.val)

/-- Represents a transformation operation on a 2x2 matrix -/
inductive Transform
| row : Unit → Transform
| col : Unit → Transform

/-- Apply a transformation to a matrix -/
def applyTransform (t : Transform) (m : Matrix2x2) : Matrix2x2 :=
  match t with
  | Transform.row _ => { a := (m.a + 1), b := (m.b + 1), c := m.c, d := m.d }
  | Transform.col _ => { a := m.a, b := m.b, c := (m.c + 1), d := (m.d + 1) }

/-- The invariant property is preserved under transformations -/
theorem invariant_preserved (m : Matrix2x2) (t : Transform) :
  invariant m = invariant (applyTransform t m) :=
sorry

/-- The 2x2 submatrices of table A and table B -/
def tableA : Matrix2x2 := { a := 19, b := 15, c := 20, d := 26 }
def tableB : Matrix2x2 := { a := 11, b := 2, c := 8, d := 5 }

/-- The invariants of table A and table B are different -/
theorem different_invariants : invariant tableA ≠ invariant tableB :=
sorry

/-- It is impossible to transform table A into table B -/
theorem transformation_impossible :
  ¬∃ (ts : List Transform), tableB = (ts.foldl (fun m t => applyTransform t m) tableA) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_preserved_different_invariants_transformation_impossible_l737_73723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l737_73757

/-- The sum of the repeating decimals 0.4̄ and 0.5̄6 is equal to 100/99 -/
theorem sum_of_repeating_decimals : 
  (∑' n : ℕ, 4 / (10 : ℚ)^(n+1)) + (∑' n : ℕ, (50 + 6/10) / (100 : ℚ)^(n+1)) = 100 / 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l737_73757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_interval_sum_l737_73776

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem max_min_interval_sum (a b : ℝ) :
  a < b ∧ 
  (∀ x ∈ Set.Icc a b, f x ≤ 1) ∧
  (∃ x ∈ Set.Icc a b, f x = 1) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ 1/3) ∧
  (∃ x ∈ Set.Icc a b, f x = 1/3) →
  a + b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_interval_sum_l737_73776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l737_73737

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : Point :=
  ⟨-Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Predicate for a point being on the asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

theorem eccentricity_range (h : Hyperbola) :
  ∃ (p : Point), on_asymptote h p ∧
    distance p (left_focus h) = 2 * distance p (right_focus h) →
  1 < eccentricity h ∧ eccentricity h ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l737_73737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_shift_l737_73714

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 3)
noncomputable def h (x : ℝ) : ℝ := 2 * Real.cos (2 * x)

-- State the theorem
theorem function_equivalence_and_shift :
  (∀ x, f x = g x) ∧
  (∀ x, g x = h (x - Real.pi / 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_and_shift_l737_73714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_example_l737_73703

/-- Converts rectangular coordinates to polar coordinates --/
noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 && y < 0 then
             2 * Real.pi - Real.arctan (abs y / x)
           else
             0  -- placeholder for other cases
  (r, θ)

/-- Theorem: The polar coordinates of (3/2, -2) are (5/2, 2π - arctan(4/3)) --/
theorem rect_to_polar_example : 
  let (r, θ) := rect_to_polar (3/2) (-2)
  r = 5/2 ∧ θ = 2 * Real.pi - Real.arctan (4/3) ∧ r ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_example_l737_73703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l737_73768

/-- Calculates the average speed of a two-part trip -/
noncomputable def averageSpeed (totalDistance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  totalDistance / (distance1 / speed1 + distance2 / speed2)

/-- Theorem: The average speed of the given trip is 32 km/h -/
theorem trip_average_speed :
  let totalDistance : ℝ := 70
  let distance1 : ℝ := 35
  let speed1 : ℝ := 48
  let distance2 : ℝ := 35
  let speed2 : ℝ := 24
  averageSpeed totalDistance distance1 speed1 distance2 speed2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l737_73768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_one_zero_l737_73733

-- Define the function F(x) as noncomputable
noncomputable def F (x : ℝ) := Real.log x - 1 / x

-- Theorem statement
theorem F_has_one_zero :
  ∃! x : ℝ, x > 0 ∧ F x = 0 :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_one_zero_l737_73733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_50_l737_73791

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 3 + a 7 = 8
  sixth_term : a 6 = 6

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first 10 terms of the given arithmetic sequence is 50 -/
theorem sum_10_is_50 (seq : ArithmeticSequence) : sum_n seq 10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_50_l737_73791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l737_73726

theorem triangle_side_length (A B : Real) (a b : Real) : 
  Real.cos A = -1/2 → B = π/4 → a = 3 → b = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l737_73726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_proposition_l737_73730

theorem quadratic_equation_proposition (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let roots := {x : ℝ | f x = 0}
  ∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧
    (¬(1 ∈ roots) ∧ 
    x + y = 2 ∧ 
    3 ∈ roots ∧ 
    x * y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_proposition_l737_73730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l737_73700

/-- The area of a triangle given its three vertices. -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The area of a triangle with vertices (0,3), (7,-1), and (2,6) is 14.5 square units. -/
theorem triangle_area_example : triangleArea 0 3 7 (-1) 2 6 = 14.5 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l737_73700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l737_73745

noncomputable section

def α : Set ℝ := {-1, 1, 1/2}

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The power function with exponent a -/
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

theorem power_function_domain_and_odd (a : ℝ) :
  a ∈ α ∧ (∀ x : ℝ, ∃ y : ℝ, y = powerFunction a x) ∧ IsOdd (powerFunction a) ↔ a = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_domain_and_odd_l737_73745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l737_73775

/-- A sequence a is a geometric sequence if there exists a common ratio r such that a_(n+1) = r * a_n for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n}, prove that if a_3 * a_4 * a_6 * a_7 = 81, then a_1 * a_9 = 9 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : IsGeometricSequence a) 
  (h_prod : a 3 * a 4 * a 6 * a 7 = 81) : a 1 * a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l737_73775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_representable_prime_l737_73738

def is_representable (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = (3^a : ℤ).sub (2^b : ℤ).natAbs

theorem smallest_non_representable_prime : 
  (∀ p : ℕ, p < 41 → Nat.Prime p → is_representable p) ∧
  Nat.Prime 41 ∧
  ¬ is_representable 41 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_representable_prime_l737_73738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangent_circles_l737_73792

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
noncomputable def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2) = radius1 + radius2

/-- The distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_tangent_circles
  (center1 center2 : ℝ × ℝ)
  (h : externally_tangent center1 center2 2 3) :
  distance center1 center2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangent_circles_l737_73792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_extra_postage_l737_73764

-- Define the envelope structure
structure Envelope where
  length : ℚ
  height : ℚ

-- Define the extra charge function
noncomputable def extraCharge (e : Envelope) : ℚ :=
  if e.length / e.height < 1.5 || e.length / e.height > 3 then 15/100 else 0

-- Define the set of envelopes
def envelopes : List Envelope := [
  { length := 7, height := 6 },
  { length := 8, height := 2 },
  { length := 7, height := 7 },
  { length := 13, height := 4 }
]

-- Theorem statement
theorem total_extra_postage :
  (envelopes.map extraCharge).sum = 60/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_extra_postage_l737_73764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l737_73725

noncomputable section

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P

-- Define the perpendicular condition
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  A.1 = -1 ∧ A.2 = P.2

-- Define the angle condition
def angle_condition (A : ℝ × ℝ) : Prop :=
  (A.2 - 0) / (1 - (-1)) = -Real.sqrt 3

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem parabola_focus_distance
  (P : ℝ × ℝ)
  (h1 : point_on_parabola P)
  (h2 : ∃ A : ℝ × ℝ, perpendicular_to_directrix P A ∧ angle_condition A) :
  distance P focus = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l737_73725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equals_negative_two_l737_73797

/-- ε₁ is a 7th root of unity -/
noncomputable def ε₁ : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

/-- The sum of cosines we want to prove -/
noncomputable def cosine_sum : ℝ :=
  1 / (2 * Real.cos (2 * Real.pi / 7)) +
  1 / (2 * Real.cos (4 * Real.pi / 7)) +
  1 / (2 * Real.cos (6 * Real.pi / 7))

/-- The main theorem -/
theorem cosine_sum_equals_negative_two : cosine_sum = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equals_negative_two_l737_73797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_transitive_l737_73716

-- Define the types for planes and lines
variable (α : Type)
variable (Line : Type)

-- Define the geometric relationships
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → α → Prop)

-- Theorem statement
theorem parallel_perpendicular_transitive 
  (a b c : Line) (plane : α) :
  parallel a b → perpendicular b plane → perpendicular c plane → parallel a c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_perpendicular_transitive_l737_73716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_ratio_l737_73765

theorem rose_ratio (total : ℕ) (red_white : ℕ) : 
  total = 80 → 
  red_white = 75 → 
  (4 * (total - red_white) : ℚ) = (total - 3 * total / 4) := 
by
  intro h_total h_red_white
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_ratio_l737_73765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_decrease_notation_l737_73759

/-- Represents temperature change in degrees Celsius -/
structure TemperatureChange where
  value : ℤ

/-- Denotes how a temperature change is represented -/
def denote (change : TemperatureChange) : ℤ := 
  if change.value > 0 then change.value else -change.value

/-- The proposition that a temperature decrease of 2°C is denoted as -2°C 
    given that a temperature rise of 1°C is denoted as +1°C -/
theorem temperature_decrease_notation 
  (h : denote ⟨1⟩ = 1) : 
  denote ⟨-2⟩ = -2 := by
  simp [denote]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_decrease_notation_l737_73759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_zero_l737_73727

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x - a - 1) / Real.log 10

-- State the theorem
theorem range_of_f_when_a_is_zero :
  ∀ y : ℝ, ∃ x : ℝ, f 0 x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_zero_l737_73727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_is_13_l737_73754

def smallest_class_size (num_perfect_scores : ℕ) (min_score : ℕ) (mean_score : ℕ) : ℕ → Prop :=
  λ n : ℕ =>
    n ≥ num_perfect_scores ∧
    (n * mean_score : ℕ) ≥ (num_perfect_scores * 100 + (n - num_perfect_scores) * min_score) ∧
    ∀ (m : ℕ), m < n →
      (m * mean_score : ℕ) < (num_perfect_scores * 100 + (m - num_perfect_scores) * min_score)

theorem smallest_class_size_is_13 :
  smallest_class_size 5 60 76 13 :=
by
  sorry

#check smallest_class_size_is_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_is_13_l737_73754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y₁_range_m_upper_bound_max_integer_m_l737_73782

noncomputable def y₁ (x₁ : ℝ) : ℝ := 4 * x₁ / (x₁^2 + 1)

def y₂ (x₂ m : ℝ) : ℝ := -x₂ + 5 - 2*m

theorem y₁_range : Set.range y₁ = Set.Icc (-2 : ℝ) 2 := by sorry

theorem m_upper_bound (h : ∀ x₁ > 0, ∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, y₂ x₂ m ≥ y₁ x₁) : 
  m ≤ 2 := by sorry

theorem max_integer_m : 
  (∃ x₂ ∈ Set.Icc (3/2 : ℝ) (9/2), y₂ x₂ m * (x₂ - 1) ≥ 1) → 
  (∀ k : ℤ, k > m → k > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y₁_range_m_upper_bound_max_integer_m_l737_73782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_A_for_quadrilateral_area_l737_73786

noncomputable section

-- Define the grid points
def K : ℝ × ℝ := (0, 1)
def O : ℝ × ℝ := (3, 0)
def Z : ℝ × ℝ := (4, 3)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Define the area of a quadrilateral given four points
noncomputable def quadArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  triangleArea p1 p2 p3 + triangleArea p1 p3 p4

-- State the theorem
theorem exists_point_A_for_quadrilateral_area :
  ∃ A : ℝ × ℝ, 
    (triangleArea K O A + triangleArea K A Z < triangleArea K O Z) ∧
    (quadArea K O Z A = 4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_A_for_quadrilateral_area_l737_73786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_dot_product_problem_l737_73715

noncomputable def m : ℝ × ℝ := (1/2, -Real.sqrt 3 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

theorem vector_problem (x : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ m = k • n x) →
  Real.tan x = -Real.sqrt 3 := by
  sorry

theorem dot_product_problem (x : ℝ) :
  m.1 * (n x).1 + m.2 * (n x).2 = 1/3 →
  0 < x ∧ x < Real.pi/2 →
  Real.cos x = (1 + 2 * Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_dot_product_problem_l737_73715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l737_73774

def sequence_a (a₁ a₂ : ℕ+) : ℕ → ℕ+
  | 0 => a₁
  | 1 => a₂
  | (n + 2) => ⟨(sequence_a a₁ a₂ n + 2017) / (1 + sequence_a a₁ a₂ (n + 1)), sorry⟩

theorem min_sum_first_two_terms :
  ∀ a₁ a₂ : ℕ+, (∀ n : ℕ, (sequence_a a₁ a₂ (n + 2)).val * (1 + (sequence_a a₁ a₂ (n + 1)).val) = (sequence_a a₁ a₂ n).val + 2017) →
  2018 ≤ a₁.val + a₂.val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l737_73774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l737_73781

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) :
  f x > 0 ↔ (1 < x ∧ x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l737_73781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l737_73758

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (1 - a) / 2 * x^2 + a * x - Real.log x

-- State the theorem
theorem function_inequality (a m : ℝ) (h_a : 4 < a ∧ a < 5) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 →
    (a - 1) / 2 * m + Real.log 2 > |f a x₁ - f a x₂|) →
  m ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l737_73758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l737_73753

open Real

def f : ℝ → ℝ := sorry

axiom f_def : ∀ x : ℝ, x ≠ 0 → f x = f 1 * x + f 2 / x - 2

theorem f_minimum_value :
  ∃ x_min : ℝ, x_min > 0 ∧ f x_min = 2 * Real.sqrt 3 - 2 ∧
  ∀ x : ℝ, x > 0 → f x ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l737_73753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l737_73784

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) + 1

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

/-- Theorem stating that (π/12, 1) is a symmetry center of g(x) -/
theorem symmetry_center_of_g :
  ∀ (x : ℝ), g (Real.pi / 12 + x) = g (Real.pi / 12 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_g_l737_73784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_5_6_l737_73780

/-- The area of a triangle with sides 5, 5, and 6 is 12 square units -/
theorem triangle_area_5_5_6 : 
  ∀ (t : Set ℝ) (a b c : ℝ),
  (∃ (x y z : ℝ), 
    (x ∈ t ∧ y ∈ t ∧ z ∈ t) ∧
    (dist x y = 5 ∧ dist y z = 5 ∧ dist z x = 6)) →
  MeasureTheory.volume t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_5_5_6_l737_73780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_integral_greater_than_trig_difference_l737_73770

theorem absolute_integral_greater_than_trig_difference : 
  ∫ x in (-1)..1, |x| > (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_integral_greater_than_trig_difference_l737_73770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tx_ty_sum_squares_l737_73789

-- Define the basic types
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Circle : Type)
variable (Line : Type)

-- Define the triangle ABC and its circumcircle ω
variable (A B C : Point) (ω : Circle)

-- Define T as the intersection of tangents
variable (T : Point)

-- Define X and Y as projections
variable (X Y : Point)

-- Define necessary functions
def dist (p q : Point) : ℝ := ‖p - q‖

-- Axioms based on the given conditions
axiom acute_scalene : True  -- Placeholder for AcuteScalene A B C
axiom circumcircle : True  -- Placeholder for Circumcircle ω A B C
axiom tangent_intersection : True  -- Placeholder for TangentIntersection ω B C T
axiom projection_X : True  -- Placeholder for Projection T X (Line.through A B)
axiom projection_Y : True  -- Placeholder for Projection T Y (Line.through A C)
axiom bt_ct_eq : dist B T = dist C T
axiom bt_value : dist B T = 20
axiom bc_value : dist B C = 30
axiom sum_squares : dist T X ^ 2 + dist T Y ^ 2 + dist X Y ^ 2 = 2193

-- Theorem to prove
theorem tx_ty_sum_squares :
  dist T X ^ 2 + dist T Y ^ 2 = 1302 :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tx_ty_sum_squares_l737_73789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l737_73767

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 1}
def B : Set ℝ := {x : ℝ | Real.rpow 2 x > 1}

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := Set.Ioo 0 1

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = open_unit_interval := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l737_73767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l737_73746

/-- The equation of a quadratic curve -/
def curve (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (-1, 6)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 4*x + y - 2 = 0

/-- Theorem: The tangent line equation is correct for the given curve and point -/
theorem tangent_line_is_correct :
  let (x₀, y₀) := point_of_tangency
  (∀ x, curve x = curve x₀ + ((2 * x₀ - 2) * (x - x₀))) →
  tangent_line x₀ y₀ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l737_73746
