import Mathlib

namespace NUMINAMATH_CALUDE_parabola_c_value_l4029_402934

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 1 = 3 →  -- vertex at (1, 3)
  p.y_at 0 = 2 →  -- passes through (0, 2)
  p.c = 2 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l4029_402934


namespace NUMINAMATH_CALUDE_area_triangle_ABP_l4029_402988

/-- Given points A and B in ℝ², and a point P on the x-axis forming a right angle with AB, 
    prove that the area of triangle ABP is 5/2. -/
theorem area_triangle_ABP (A B P : ℝ × ℝ) : 
  A = (1, 1) →
  B = (2, -1) →
  P.2 = 0 →  -- P is on the x-axis
  (P.1 - B.1) * (B.1 - A.1) + (P.2 - B.2) * (B.2 - A.2) = 0 →  -- ∠ABP = 90°
  abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2)) / 2 = 5/2 := by
sorry


end NUMINAMATH_CALUDE_area_triangle_ABP_l4029_402988


namespace NUMINAMATH_CALUDE_blueberry_baskets_l4029_402997

theorem blueberry_baskets (initial_berries : ℕ) (total_berries : ℕ) : 
  initial_berries = 20 →
  total_berries = 200 →
  (total_berries / initial_berries) - 1 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_blueberry_baskets_l4029_402997


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_square_le_power_two_l4029_402970

theorem negation_of_proposition (p : ℕ → Prop) :
  (¬∀ n : ℕ, p n) ↔ (∃ n : ℕ, ¬p n) := by sorry

theorem negation_of_square_le_power_two :
  (¬∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_square_le_power_two_l4029_402970


namespace NUMINAMATH_CALUDE_rectangle_area_l4029_402943

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 →
  ratio = 3 →
  (2 * r * ratio) * (2 * r) = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4029_402943


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l4029_402914

theorem white_surface_area_fraction (cube_edge : ℕ) (total_unit_cubes : ℕ) 
  (white_unit_cubes : ℕ) (black_unit_cubes : ℕ) :
  cube_edge = 4 →
  total_unit_cubes = 64 →
  white_unit_cubes = 48 →
  black_unit_cubes = 16 →
  white_unit_cubes + black_unit_cubes = total_unit_cubes →
  (black_unit_cubes : ℚ) * 3 / (6 * cube_edge^2 : ℚ) = 1/2 →
  (6 * cube_edge^2 - black_unit_cubes * 3 : ℚ) / (6 * cube_edge^2 : ℚ) = 1/2 := by
  sorry

#check white_surface_area_fraction

end NUMINAMATH_CALUDE_white_surface_area_fraction_l4029_402914


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l4029_402907

/-- Given a total number of candies and classmates, calculates the maximum number
    of candies Vovochka can keep while satisfying the distribution condition. -/
def max_candies_for_vovochka (total_candies : ℕ) (num_classmates : ℕ) : ℕ :=
  total_candies - (num_classmates - 1) * 7 + 4

/-- Checks if the candy distribution satisfies the condition that
    any 16 classmates have at least 100 candies. -/
def satisfies_condition (candies_kept : ℕ) (total_candies : ℕ) (num_classmates : ℕ) : Prop :=
  ∀ (group : Finset (Fin num_classmates)),
    group.card = 16 →
    (total_candies - candies_kept) * 16 / num_classmates ≥ 100

theorem vovochka_candy_theorem (total_candies num_classmates : ℕ)
    (h1 : total_candies = 200)
    (h2 : num_classmates = 25) :
    let max_candies := max_candies_for_vovochka total_candies num_classmates
    satisfies_condition max_candies total_candies num_classmates ∧
    ∀ k, k > max_candies →
      ¬satisfies_condition k total_candies num_classmates :=
  sorry

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l4029_402907


namespace NUMINAMATH_CALUDE_bouncyBallCostDifferenceIs24_l4029_402940

/-- Calculates the difference in cost between red bouncy balls and the combined cost of yellow and blue bouncy balls -/
def bouncyBallCostDifference : ℚ :=
  let redPacks : ℕ := 5
  let yellowPacks : ℕ := 4
  let bluePacks : ℕ := 3
  let redBallsPerPack : ℕ := 18
  let yellowBallsPerPack : ℕ := 15
  let blueBallsPerPack : ℕ := 12
  let redBallPrice : ℚ := 3/2
  let yellowBallPrice : ℚ := 5/4
  let blueBallPrice : ℚ := 1
  let redCost := (redPacks * redBallsPerPack : ℕ) * redBallPrice
  let yellowCost := (yellowPacks * yellowBallsPerPack : ℕ) * yellowBallPrice
  let blueCost := (bluePacks * blueBallsPerPack : ℕ) * blueBallPrice
  redCost - (yellowCost + blueCost)

theorem bouncyBallCostDifferenceIs24 : bouncyBallCostDifference = 24 := by
  sorry

end NUMINAMATH_CALUDE_bouncyBallCostDifferenceIs24_l4029_402940


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l4029_402913

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 = 10) :
  a 2 = 5 :=
sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l4029_402913


namespace NUMINAMATH_CALUDE_boat_travel_time_l4029_402981

/-- Proves that a boat traveling upstream for 1.5 hours will take 1 hour to travel the same distance downstream, given the boat's speed in still water and the stream's speed. -/
theorem boat_travel_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (upstream_time : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : stream_speed = 3) 
  (h3 : upstream_time = 1.5) : 
  (boat_speed - stream_speed) * upstream_time / (boat_speed + stream_speed) = 1 := by
  sorry

#check boat_travel_time

end NUMINAMATH_CALUDE_boat_travel_time_l4029_402981


namespace NUMINAMATH_CALUDE_value_range_of_sum_product_l4029_402968

theorem value_range_of_sum_product (x : ℝ) : 
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
sorry

end NUMINAMATH_CALUDE_value_range_of_sum_product_l4029_402968


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l4029_402939

theorem laptop_sticker_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 50 = 0.7 * sticker_price + 30) → 
  sticker_price = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l4029_402939


namespace NUMINAMATH_CALUDE_int_part_one_plus_sqrt_seven_l4029_402948

theorem int_part_one_plus_sqrt_seven : ⌊1 + Real.sqrt 7⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_int_part_one_plus_sqrt_seven_l4029_402948


namespace NUMINAMATH_CALUDE_money_never_equal_l4029_402923

/-- Represents the amount of money in Kiriels and Dariels -/
structure Money where
  kiriels : ℕ
  dariels : ℕ

/-- Represents a currency exchange operation -/
inductive Exchange
  | KirielToDariel : ℕ → Exchange
  | DarielToKiriel : ℕ → Exchange

/-- Applies a single exchange operation to a Money value -/
def applyExchange (m : Money) (e : Exchange) : Money :=
  match e with
  | Exchange.KirielToDariel n => 
      ⟨m.kiriels - n, m.dariels + 10 * n⟩
  | Exchange.DarielToKiriel n => 
      ⟨m.kiriels + 10 * n, m.dariels - n⟩

/-- Applies a sequence of exchanges to an initial Money value -/
def applyExchanges (initial : Money) : List Exchange → Money
  | [] => initial
  | e :: es => applyExchanges (applyExchange initial e) es

theorem money_never_equal :
  ∀ (exchanges : List Exchange),
    let final := applyExchanges ⟨0, 1⟩ exchanges
    final.kiriels ≠ final.dariels :=
  sorry


end NUMINAMATH_CALUDE_money_never_equal_l4029_402923


namespace NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l4029_402977

theorem positive_sum_from_positive_difference (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l4029_402977


namespace NUMINAMATH_CALUDE_function_properties_l4029_402950

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.sin x + a

theorem function_properties (t : ℝ) :
  (∃ a : ℝ, f a π = 1 ∧ f a t = 2) →
  (∃ a : ℝ, a = 1 ∧ f a (-t) = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4029_402950


namespace NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l4029_402916

/-- 
Given integers x, y, z, u forming an arithmetic progression and satisfying x^3 + y^3 + z^3 = u^3,
prove that there exists an integer d such that x = 3d, y = 4d, z = 5d, and u = 6d.
-/
theorem arithmetic_progression_cube_sum (x y z u : ℤ) 
  (h_arith_prog : ∃ (d : ℤ), y = x + d ∧ z = y + d ∧ u = z + d)
  (h_cube_sum : x^3 + y^3 + z^3 = u^3) :
  ∃ (d : ℤ), x = 3*d ∧ y = 4*d ∧ z = 5*d ∧ u = 6*d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cube_sum_l4029_402916


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l4029_402960

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + k ≠ 0) → k > 25/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l4029_402960


namespace NUMINAMATH_CALUDE_hope_project_donation_proof_l4029_402904

theorem hope_project_donation_proof 
  (total_donation_A total_donation_B : ℝ)
  (donation_difference : ℝ)
  (people_ratio : ℝ) :
  total_donation_A = 20000 →
  total_donation_B = 20000 →
  donation_difference = 20 →
  people_ratio = 4/5 →
  ∃ (people_A : ℝ) (donation_A donation_B : ℝ),
    people_A > 0 ∧
    donation_A > 0 ∧
    donation_B > 0 ∧
    people_A * donation_A = total_donation_A ∧
    (people_ratio * people_A) * donation_B = total_donation_B ∧
    donation_B = donation_A + donation_difference ∧
    donation_A = 80 ∧
    donation_B = 100 :=
by sorry

end NUMINAMATH_CALUDE_hope_project_donation_proof_l4029_402904


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l4029_402982

/-- If z = (2i-1)/i is a complex number with real part a and imaginary part b, then ab = 2 -/
theorem complex_product_real_imag_parts : 
  let z : ℂ := (2 * Complex.I - 1) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l4029_402982


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4029_402941

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l4029_402941


namespace NUMINAMATH_CALUDE_smallest_battleship_board_l4029_402920

/-- Represents a ship in the Battleship game -/
structure Ship :=
  (size : Nat)

/-- Represents the Battleship game board -/
structure Board :=
  (size : Nat)
  (ships : List Ship)

/-- Checks if the given board configuration is valid -/
def isValidBoard (board : Board) : Prop :=
  board.size ≥ 7 ∧
  board.ships.length = 10 ∧
  (board.ships.filter (λ s => s.size = 4)).length = 1 ∧
  (board.ships.filter (λ s => s.size = 3)).length = 2 ∧
  (board.ships.filter (λ s => s.size = 2)).length = 3 ∧
  (board.ships.filter (λ s => s.size = 1)).length = 4

/-- Theorem: The smallest valid square board for Battleship is 7x7 -/
theorem smallest_battleship_board :
  ∀ (board : Board), isValidBoard board →
    ∃ (minBoard : Board), isValidBoard minBoard ∧ minBoard.size = 7 ∧
      ∀ (b : Board), isValidBoard b → b.size ≥ minBoard.size :=
by sorry

end NUMINAMATH_CALUDE_smallest_battleship_board_l4029_402920


namespace NUMINAMATH_CALUDE_bicycle_sale_profit_percentage_l4029_402902

/-- Calculates the profit percentage of seller A given the conditions of the bicycle sale problem -/
theorem bicycle_sale_profit_percentage 
  (cp_a : ℝ)     -- Cost price for A
  (sp_c : ℝ)     -- Selling price for C
  (profit_b : ℝ) -- Profit percentage for B
  (h1 : cp_a = 120)
  (h2 : sp_c = 225)
  (h3 : profit_b = 25) :
  (((sp_c / (1 + profit_b / 100) - cp_a) / cp_a) * 100 = 50) := by
  sorry

#check bicycle_sale_profit_percentage

end NUMINAMATH_CALUDE_bicycle_sale_profit_percentage_l4029_402902


namespace NUMINAMATH_CALUDE_remainder_theorem_l4029_402918

theorem remainder_theorem (x : ℝ) : 
  let R := fun x => 3^125 * x - 2^125 * x + 2^125 - 2 * 3^125
  let divisor := fun x => x^2 - 5*x + 6
  ∃ Q : ℝ → ℝ, x^125 = Q x * divisor x + R x ∧ 
  (∀ a b : ℝ, R x = a * x + b → (a = 3^125 - 2^125 ∧ b = 2^125 - 2 * 3^125)) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4029_402918


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_m_l4029_402975

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem 1: Solving the inequality f(x) > 3-4x
theorem solve_inequality : 
  ∀ x : ℝ, f x > 3 - 4*x ↔ x > 3/5 := by sorry

-- Theorem 2: Finding the range of m
theorem range_of_m : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) ↔ m ∈ Set.Icc (-1/6 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_m_l4029_402975


namespace NUMINAMATH_CALUDE_total_beverage_amount_l4029_402919

/-- Given 5 bottles, each containing 242.7 ml of beverage, 
    the total amount of beverage is 1213.5 ml. -/
theorem total_beverage_amount :
  let num_bottles : ℕ := 5
  let amount_per_bottle : ℝ := 242.7
  num_bottles * amount_per_bottle = 1213.5 := by
sorry

end NUMINAMATH_CALUDE_total_beverage_amount_l4029_402919


namespace NUMINAMATH_CALUDE_value_of_x_l4029_402974

theorem value_of_x (x y z : ℚ) : 
  x = (1/2) * y → 
  y = (1/5) * z → 
  z = 60 → 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l4029_402974


namespace NUMINAMATH_CALUDE_new_plant_characteristics_l4029_402987

/-- Represents a plant with genetic characteristics -/
structure Plant where
  ploidy : Nat
  has_homologous_chromosomes : Bool
  can_form_fertile_gametes : Bool
  homozygosity : Option Bool

/-- Represents the process of obtaining new plants from treated corn -/
def obtain_new_plants (original : Plant) (colchicine_treated : Bool) (anther_culture : Bool) : Plant :=
  sorry

/-- Theorem stating the characteristics of new plants obtained from treated corn -/
theorem new_plant_characteristics 
  (original : Plant)
  (h_original_diploid : original.ploidy = 2)
  (h_colchicine_treated : Bool)
  (h_anther_culture : Bool) :
  let new_plant := obtain_new_plants original h_colchicine_treated h_anther_culture
  new_plant.ploidy = 1 ∧ 
  new_plant.has_homologous_chromosomes = true ∧
  new_plant.can_form_fertile_gametes = true ∧
  new_plant.homozygosity = none :=
by sorry

end NUMINAMATH_CALUDE_new_plant_characteristics_l4029_402987


namespace NUMINAMATH_CALUDE_consecutive_points_ratio_l4029_402945

/-- Given five consecutive points on a line, prove the ratio of distances -/
theorem consecutive_points_ratio (a b c d e : ℝ) : 
  (b - a = 5) → 
  (c - a = 11) → 
  (e - d = 7) → 
  (e - a = 20) → 
  (c - b) / (d - c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_ratio_l4029_402945


namespace NUMINAMATH_CALUDE_min_value_on_line_l4029_402925

/-- The minimum value of 9^x + 3^y where (x, y) is on the line y = 4 - 2x -/
theorem min_value_on_line : ∃ (min : ℝ),
  (∀ (x y : ℝ), y = 4 - 2*x → 9^x + 3^y ≥ min) ∧
  (∃ (x y : ℝ), y = 4 - 2*x ∧ 9^x + 3^y = min) ∧
  min = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_l4029_402925


namespace NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l4029_402954

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * x - 1) * (x - b)

-- State the theorem
theorem solution_set_of_f_neg_x (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, f a b (-x) < 0 ↔ x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_neg_x_l4029_402954


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_sum_of_three_numbers_proof_l4029_402966

theorem sum_of_three_numbers : ℕ → ℕ → ℕ → Prop :=
  fun second first third =>
    first = 2 * second ∧
    third = first / 3 ∧
    second = 60 →
    first + second + third = 220

-- The proof is omitted
theorem sum_of_three_numbers_proof : sum_of_three_numbers 60 120 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_sum_of_three_numbers_proof_l4029_402966


namespace NUMINAMATH_CALUDE_bus_rental_solution_l4029_402994

/-- Represents the bus rental problem for a school study tour. -/
structure BusRentalProblem where
  capacityA : Nat  -- Capacity of A type bus
  capacityB : Nat  -- Capacity of B type bus
  extraPeople : Nat  -- People without seats in original plan
  fewerBusesB : Nat  -- Number of fewer B type buses needed
  totalBuses : Nat  -- Total number of buses to be rented
  maxTypeB : Nat  -- Maximum number of B type buses
  feeA : Nat  -- Rental fee for A type bus
  feeB : Nat  -- Rental fee for B type bus

/-- Represents a bus rental scheme. -/
structure RentalScheme where
  numA : Nat  -- Number of A type buses
  numB : Nat  -- Number of B type buses

/-- The main theorem about the bus rental problem. -/
theorem bus_rental_solution (p : BusRentalProblem)
  (h1 : p.capacityA = 45)
  (h2 : p.capacityB = 60)
  (h3 : p.extraPeople = 30)
  (h4 : p.fewerBusesB = 6)
  (h5 : p.totalBuses = 25)
  (h6 : p.maxTypeB = 7)
  (h7 : p.feeA = 220)
  (h8 : p.feeB = 300) :
  ∃ (originalA totalPeople : Nat) (schemes : List RentalScheme) (bestScheme : RentalScheme),
    originalA = 26 ∧
    totalPeople = 1200 ∧
    schemes = [⟨20, 5⟩, ⟨19, 6⟩, ⟨18, 7⟩] ∧
    bestScheme = ⟨20, 5⟩ ∧
    (∀ scheme ∈ schemes, 
      scheme.numA + scheme.numB = p.totalBuses ∧
      scheme.numB ≤ p.maxTypeB ∧
      scheme.numA * p.capacityA + scheme.numB * p.capacityB ≥ totalPeople) ∧
    (∀ scheme ∈ schemes,
      scheme.numA * p.feeA + scheme.numB * p.feeB ≥ 
      bestScheme.numA * p.feeA + bestScheme.numB * p.feeB) := by
  sorry

end NUMINAMATH_CALUDE_bus_rental_solution_l4029_402994


namespace NUMINAMATH_CALUDE_chemical_representations_correct_l4029_402900

/-- Represents a chemical element -/
inductive Element : Type
| C : Element
| H : Element
| O : Element
| N : Element
| Si : Element
| P : Element

/-- Represents a chemical formula -/
structure ChemicalFormula :=
  (elements : List (Element × ℕ))

/-- Represents a structural formula -/
structure StructuralFormula :=
  (formula : String)

/-- Definition of starch chemical formula -/
def starchFormula : ChemicalFormula :=
  ⟨[(Element.C, 6), (Element.H, 10), (Element.O, 5)]⟩

/-- Definition of glycine structural formula -/
def glycineFormula : StructuralFormula :=
  ⟨"H₂N-CH₂-COOH"⟩

/-- Definition of silicate-containing materials -/
def silicateProducts : List String :=
  ["glass", "ceramics", "cement"]

/-- Definition of red tide causing elements -/
def redTideElements : List Element :=
  [Element.N, Element.P]

/-- Theorem stating the correctness of the chemical representations -/
theorem chemical_representations_correct :
  (starchFormula.elements = [(Element.C, 6), (Element.H, 10), (Element.O, 5)]) ∧
  (glycineFormula.formula = "H₂N-CH₂-COOH") ∧
  (∀ product ∈ silicateProducts, ∃ e ∈ product.toList, e = 'S') ∧
  (redTideElements = [Element.N, Element.P]) :=
sorry


end NUMINAMATH_CALUDE_chemical_representations_correct_l4029_402900


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l4029_402951

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 36 →
  crossing_time = 24.198064154867613 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 131.98064154867613 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l4029_402951


namespace NUMINAMATH_CALUDE_rectangle_area_l4029_402938

theorem rectangle_area : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x * y = (x + 3) * (y - 1) ∧
  x * y = (x - 4) * (y + 1.5) ∧
  x * y = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4029_402938


namespace NUMINAMATH_CALUDE_existence_of_m_n_l4029_402984

theorem existence_of_m_n (p s : ℕ) (hp : Nat.Prime p) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧
    (m * s % p : ℚ) / p < (n * s % p : ℚ) / p ∧ (n * s % p : ℚ) / p < (s : ℚ) / p) ↔
  ¬(s ∣ p - 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l4029_402984


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l4029_402922

theorem initial_number_of_persons (N : ℕ) 
  (h1 : ∃ (avg : ℝ), N * (avg + 5) - N * avg = 105 - 65) : N = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l4029_402922


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l4029_402958

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 120) (h₃ : a₃ = 27/16) :
  ∃ b : ℝ, b > 0 ∧ b * b = a₁ * a₃ ∧ b = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l4029_402958


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4029_402973

def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4029_402973


namespace NUMINAMATH_CALUDE_rectangle_b_values_product_l4029_402952

theorem rectangle_b_values_product (b₁ b₂ : ℝ) : 
  (b₁ ≠ b₂) →
  (abs (b₁ - 2) = 6 ∨ abs (b₂ - 2) = 6) →
  (abs (b₁ - 2) = 6 → b₂ = 2 - (b₁ - 2)) →
  (abs (b₂ - 2) = 6 → b₁ = 2 - (b₂ - 2)) →
  b₁ * b₂ = -32 := by
sorry

end NUMINAMATH_CALUDE_rectangle_b_values_product_l4029_402952


namespace NUMINAMATH_CALUDE_circle_through_points_center_on_line_l4029_402978

/-- A circle passing through two points with its center on a given line -/
theorem circle_through_points_center_on_line (A B O : ℝ × ℝ) (r : ℝ) :
  A = (1, -1) →
  B = (-1, 1) →
  O.1 + O.2 = 2 →
  r = 2 →
  ∀ (x y : ℝ), (x - O.1)^2 + (y - O.2)^2 = r^2 ↔
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∨
    (x - O.1)^2 + (y - O.2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_center_on_line_l4029_402978


namespace NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l4029_402933

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ 16 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_c_and_d_l4029_402933


namespace NUMINAMATH_CALUDE_ms_leech_class_boys_l4029_402912

/-- Proves that the number of boys in Ms. Leech's class is 10 -/
theorem ms_leech_class_boys (total_students : ℕ) (total_cups : ℕ) (cups_per_boy : ℕ) :
  total_students = 30 →
  total_cups = 90 →
  cups_per_boy = 5 →
  ∃ (boys : ℕ),
    boys * 3 = total_students ∧
    boys * cups_per_boy = total_cups / 2 ∧
    boys = 10 :=
by sorry

end NUMINAMATH_CALUDE_ms_leech_class_boys_l4029_402912


namespace NUMINAMATH_CALUDE_max_sum_of_three_integers_with_product_24_l4029_402996

theorem max_sum_of_three_integers_with_product_24 :
  (∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧
    ∀ (x y z : ℕ+), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 24 →
      x + y + z ≤ a + b + c) ∧
  (∀ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 →
    a + b + c ≤ 15) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_integers_with_product_24_l4029_402996


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l4029_402911

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 140 →
  red_students = 60 →
  green_students = 80 →
  total_pairs = 70 →
  red_red_pairs = 10 →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 20 ∧ 
    green_green_pairs + red_red_pairs + (total_pairs - green_green_pairs - red_red_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l4029_402911


namespace NUMINAMATH_CALUDE_product_plus_twenty_l4029_402928

theorem product_plus_twenty : ∃ n : ℕ, n = 5 * 7 ∧ n + 12 + 8 = 55 := by sorry

end NUMINAMATH_CALUDE_product_plus_twenty_l4029_402928


namespace NUMINAMATH_CALUDE_slower_train_speed_l4029_402992

/-- Prove that the speed of the slower train is 36 km/hr -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 55) 
  (h2 : faster_speed = 47) 
  (h3 : passing_time = 36) : 
  ∃ (slower_speed : ℝ), 
    slower_speed = 36 ∧ 
    (2 * train_length) = (faster_speed - slower_speed) * (5/18) * passing_time :=
sorry

end NUMINAMATH_CALUDE_slower_train_speed_l4029_402992


namespace NUMINAMATH_CALUDE_inequality_preservation_l4029_402915

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l4029_402915


namespace NUMINAMATH_CALUDE_jack_final_position_l4029_402930

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack goes up -/
def flights_up : ℕ := 3

/-- Represents the number of flights Jack goes down -/
def flights_down : ℕ := 6

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Theorem stating that Jack ends up 24 feet further down than his starting point -/
theorem jack_final_position : 
  (flights_down - flights_up) * steps_per_flight * step_height / inches_per_foot = 24 := by
  sorry


end NUMINAMATH_CALUDE_jack_final_position_l4029_402930


namespace NUMINAMATH_CALUDE_towel_average_price_l4029_402990

theorem towel_average_price :
  let towel_group1 : ℕ := 3
  let price1 : ℕ := 100
  let towel_group2 : ℕ := 5
  let price2 : ℕ := 150
  let towel_group3 : ℕ := 2
  let price3 : ℕ := 400
  let total_towels := towel_group1 + towel_group2 + towel_group3
  let total_cost := towel_group1 * price1 + towel_group2 * price2 + towel_group3 * price3
  (total_cost : ℚ) / (total_towels : ℚ) = 185 := by
  sorry

end NUMINAMATH_CALUDE_towel_average_price_l4029_402990


namespace NUMINAMATH_CALUDE_stacy_height_proof_l4029_402998

def stacy_height_problem (last_year_height : ℕ) (brother_growth : ℕ) (growth_difference : ℕ) : Prop :=
  let stacy_growth : ℕ := brother_growth + growth_difference
  let current_height : ℕ := last_year_height + stacy_growth
  current_height = 57

theorem stacy_height_proof :
  stacy_height_problem 50 1 6 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_proof_l4029_402998


namespace NUMINAMATH_CALUDE_ellipse_conditions_l4029_402965

/-- A curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate for a curve being an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

/-- The conditions a > 0 and b > 0 -/
def positive_conditions (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0

theorem ellipse_conditions (c : Curve) :
  (positive_conditions c → is_ellipse c) ∧
  ¬(is_ellipse c → positive_conditions c) :=
sorry

end NUMINAMATH_CALUDE_ellipse_conditions_l4029_402965


namespace NUMINAMATH_CALUDE_quadratic_function_range_l4029_402963

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

/-- The range of f is [0, +∞) -/
def has_range_zero_to_infinity (m : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f m x = y

theorem quadratic_function_range (m : ℝ) :
  has_range_zero_to_infinity m → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l4029_402963


namespace NUMINAMATH_CALUDE_alarm_clock_probability_l4029_402962

theorem alarm_clock_probability (A B : ℝ) (hA : A = 0.80) (hB : B = 0.90) :
  1 - (1 - A) * (1 - B) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_alarm_clock_probability_l4029_402962


namespace NUMINAMATH_CALUDE_bruce_mangoes_l4029_402953

/-- Calculates the quantity of mangoes purchased given the total amount paid, grape quantity, grape price, and mango price -/
def mangoes_purchased (total_paid : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  ((total_paid - grape_quantity * grape_price) / mango_price : ℕ)

/-- Theorem stating that given the problem conditions, Bruce purchased 9 kg of mangoes -/
theorem bruce_mangoes :
  mangoes_purchased 985 7 70 55 = 9 := by
sorry

end NUMINAMATH_CALUDE_bruce_mangoes_l4029_402953


namespace NUMINAMATH_CALUDE_higher_selling_price_is_360_l4029_402957

/-- The higher selling price of an article, given its cost and profit conditions -/
def higherSellingPrice (cost : ℚ) (lowerPrice : ℚ) : ℚ :=
  let profitAtLowerPrice := lowerPrice - cost
  let additionalProfit := (5 / 100) * cost
  cost + profitAtLowerPrice + additionalProfit

/-- Theorem stating that the higher selling price is 360, given the conditions -/
theorem higher_selling_price_is_360 :
  higherSellingPrice 400 340 = 360 := by
  sorry

#eval higherSellingPrice 400 340

end NUMINAMATH_CALUDE_higher_selling_price_is_360_l4029_402957


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_l4029_402976

def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic 2 \ {2}) ∪ Set.Ici 3

theorem fractional_inequality_solution :
  {x : ℝ | (x - 3) / (x - 2) ≥ 0} = {x : ℝ | solution_set x} :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_l4029_402976


namespace NUMINAMATH_CALUDE_thomas_monthly_earnings_l4029_402985

/-- Calculates Thomas's total earnings for one month --/
def thomasEarnings (initialWage : ℝ) (weeklyIncrease : ℝ) (overtimeHours : ℕ) (overtimeRate : ℝ) (deduction : ℝ) : ℝ :=
  let week1 := initialWage
  let week2 := initialWage * (1 + weeklyIncrease)
  let week3 := week2 * (1 + weeklyIncrease)
  let week4 := week3 * (1 + weeklyIncrease)
  let overtimePay := (overtimeHours : ℝ) * overtimeRate
  week1 + week2 + week3 + week4 + overtimePay - deduction

/-- Theorem stating that Thomas's earnings for the month equal $19,761.07 --/
theorem thomas_monthly_earnings :
  thomasEarnings 4550 0.05 10 25 100 = 19761.07 := by
  sorry

end NUMINAMATH_CALUDE_thomas_monthly_earnings_l4029_402985


namespace NUMINAMATH_CALUDE_vector_perpendicular_and_obtuse_angle_l4029_402983

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define x and y as functions of k
def x (k : ℝ) : Fin 2 → ℝ := ![k - 3, 2*k + 2]
def y : Fin 2 → ℝ := ![10, -4]

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the theorem
theorem vector_perpendicular_and_obtuse_angle (k : ℝ) :
  (dot_product (x k) y = 0 ↔ k = 19) ∧
  (dot_product (x k) y < 0 ↔ k < 19 ∧ k ≠ -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_and_obtuse_angle_l4029_402983


namespace NUMINAMATH_CALUDE_circle_center_l4029_402926

/-- The equation of a circle in the form (x - h)² + (y - k)² = r², 
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenCircle : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center : 
  ∃ r : ℝ, ∀ x y : ℝ, GivenCircle x y ↔ CircleEquation 2 (-3) r x y :=
sorry

end NUMINAMATH_CALUDE_circle_center_l4029_402926


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l4029_402971

def is_geometric_progression (a : ℝ) (r : ℝ) : List ℝ → Prop
  | [x₁, x₂, x₃, x₄, x₅] => x₁ = a ∧ x₂ = a * r ∧ x₃ = a * r^2 ∧ x₄ = a * r^3 ∧ x₅ = a * r^4
  | _ => False

theorem geometric_progression_problem (a r : ℝ) (h₁ : a + a * r^2 + a * r^4 = 63) (h₂ : a * r + a * r^3 = 30) :
  is_geometric_progression a r [3, 6, 12, 24, 48] ∨ is_geometric_progression a r [48, 24, 12, 6, 3] := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l4029_402971


namespace NUMINAMATH_CALUDE_parabola_properties_l4029_402999

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  h : eq = fun x ↦ a * x^2 + 2 * a * x - 1

/-- Points on the parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.eq x = y

/-- Axis of symmetry -/
def AxisOfSymmetry (p : Parabola) : ℝ := -1

/-- Vertex on x-axis condition -/
def VertexOnXAxis (p : Parabola) : Prop :=
  p.a = -1

theorem parabola_properties (p : Parabola) (m y₁ y₂ : ℝ) 
  (hM : PointOnParabola p m y₁) 
  (hN : PointOnParabola p 2 y₂)
  (h_y : y₁ > y₂) :
  (AxisOfSymmetry p = -1) ∧
  (VertexOnXAxis p → p.eq = fun x ↦ -x^2 - 2*x - 1) ∧
  ((p.a > 0 → (m > 2 ∨ m < -4)) ∧ 
   (p.a < 0 → (-4 < m ∧ m < 2))) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l4029_402999


namespace NUMINAMATH_CALUDE_vector_BC_l4029_402989

def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (4, 5)

theorem vector_BC : (C.1 - B.1, C.2 - B.2) = (3, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l4029_402989


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_find_m_value_l4029_402917

-- Part 1
theorem simplify_and_evaluate (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b) = 5 := by sorry

-- Part 2
theorem find_m_value (a b m : ℚ) :
  (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = -3*b^2 → m = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_find_m_value_l4029_402917


namespace NUMINAMATH_CALUDE_kite_only_always_perpendicular_diagonals_l4029_402935

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Rhombus
  | Rectangle
  | Square
  | Kite
  | IsoscelesTrapezoid

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Kite => true
  | _ => false

-- Theorem statement
theorem kite_only_always_perpendicular_diagonals :
  ∀ q : Quadrilateral, has_perpendicular_diagonals q ↔ q = Quadrilateral.Kite :=
by sorry

end NUMINAMATH_CALUDE_kite_only_always_perpendicular_diagonals_l4029_402935


namespace NUMINAMATH_CALUDE_water_addition_theorem_l4029_402921

/-- Represents the amount of water that can be added to the 6-liter bucket --/
def water_to_add (bucket3 bucket5 bucket6 : ℕ) : ℕ :=
  bucket6 - (bucket5 - bucket3)

/-- Theorem stating the amount of water that can be added to the 6-liter bucket --/
theorem water_addition_theorem :
  water_to_add 3 5 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_water_addition_theorem_l4029_402921


namespace NUMINAMATH_CALUDE_square_area_ratio_l4029_402947

theorem square_area_ratio (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l4029_402947


namespace NUMINAMATH_CALUDE_total_envelopes_is_975_l4029_402927

/-- The number of envelopes Kiera has of each color and in total -/
structure EnvelopeCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ
  total : ℕ

/-- Calculates the total number of envelopes given the conditions -/
def calculateEnvelopes : EnvelopeCount :=
  let blue := 120
  let yellow := blue - 25
  let green := 5 * yellow
  let red := (blue + yellow) / 2
  let purple := red + 71
  let total := blue + yellow + green + red + purple
  { blue := blue
  , yellow := yellow
  , green := green
  , red := red
  , purple := purple
  , total := total }

/-- Theorem stating that the total number of envelopes is 975 -/
theorem total_envelopes_is_975 : calculateEnvelopes.total = 975 := by
  sorry

#eval calculateEnvelopes.total

end NUMINAMATH_CALUDE_total_envelopes_is_975_l4029_402927


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4029_402964

/-- Represents a hyperbola with equation x^2 - y^2/3 = 1 -/
def Hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/3 = 1}

/-- The equation of asymptotes for the given hyperbola -/
def AsymptoteEquation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Theorem stating that the given equation represents the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola → (AsymptoteEquation x y ↔ (x, y) ∈ closure Hyperbola \ Hyperbola) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4029_402964


namespace NUMINAMATH_CALUDE_count_true_propositions_l4029_402908

theorem count_true_propositions : 
  (∀ (x : ℝ), x^2 + 1 > 0) ∧ 
  (∃ (x : ℤ), x^3 < 1) ∧ 
  (∀ (x : ℚ), x^2 ≠ 2) ∧ 
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

#check count_true_propositions

end NUMINAMATH_CALUDE_count_true_propositions_l4029_402908


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4029_402979

theorem perfect_square_condition (a : ℕ+) 
  (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ (n^2 * a.val - 1) % d = 0) : 
  ∃ k : ℕ, a.val = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4029_402979


namespace NUMINAMATH_CALUDE_complement_A_in_U_l4029_402936

def U : Set ℕ := {x | (x + 1) / (x - 5) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {0, 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l4029_402936


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4029_402995

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + 2 * i) / i = b + i →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4029_402995


namespace NUMINAMATH_CALUDE_rd_investment_exceeds_target_l4029_402929

def initial_investment : ℝ := 1.3
def annual_increase : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015
def target_year : ℕ := 2019

theorem rd_investment_exceeds_target :
  (initial_investment * (1 + annual_increase) ^ (target_year - start_year) > target_investment) ∧
  (∀ y : ℕ, y < target_year → initial_investment * (1 + annual_increase) ^ (y - start_year) ≤ target_investment) :=
by sorry

end NUMINAMATH_CALUDE_rd_investment_exceeds_target_l4029_402929


namespace NUMINAMATH_CALUDE_axiom_1_l4029_402901

-- Define the types for points, lines, and planes
variable {Point Line Plane : Type}

-- Define the relations for points being on lines and planes
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_on_plane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (l : Line) (α : Plane) :
  (∃ (A B : Point), on_line A l ∧ on_line B l ∧ on_plane A α ∧ on_plane B α) →
  line_on_plane l α :=
sorry

end NUMINAMATH_CALUDE_axiom_1_l4029_402901


namespace NUMINAMATH_CALUDE_expected_cases_correct_l4029_402910

/-- The probability of an American having the disease -/
def disease_probability : ℚ := 1 / 3

/-- The total number of Americans in the sample -/
def sample_size : ℕ := 450

/-- The expected number of Americans with the disease in the sample -/
def expected_cases : ℕ := 150

/-- Theorem stating that the expected number of cases is correct -/
theorem expected_cases_correct : 
  ↑expected_cases = ↑sample_size * disease_probability := by sorry

end NUMINAMATH_CALUDE_expected_cases_correct_l4029_402910


namespace NUMINAMATH_CALUDE_johns_number_l4029_402980

theorem johns_number (n : ℕ) : 
  (125 ∣ n) ∧ 
  (180 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n ≤ 3000 ∧
  (∀ m : ℕ, (125 ∣ m) ∧ (180 ∣ m) ∧ 1000 ≤ m ∧ m ≤ 3000 → n ≤ m) →
  n = 1800 := by
sorry

end NUMINAMATH_CALUDE_johns_number_l4029_402980


namespace NUMINAMATH_CALUDE_sin_function_smallest_c_l4029_402956

/-- 
Given a sinusoidal function f(x) = a * sin(b * x + c) where a, b, and c are positive constants,
if f(x) reaches its maximum at x = 0, then the smallest possible value of c is π/2.
-/
theorem sin_function_smallest_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin c) → c = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_function_smallest_c_l4029_402956


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l4029_402991

/-- Given that (x^2 - 1) + (x^2 + 3x + 2)i is a purely imaginary number, prove that x = 1 -/
theorem purely_imaginary_complex_number (x : ℝ) : 
  (x^2 - 1 : ℂ) + (x^2 + 3*x + 2 : ℂ)*I = (0 : ℂ) + (y : ℝ)*I ∧ y ≠ 0 → x = 1 := by
  sorry


end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l4029_402991


namespace NUMINAMATH_CALUDE_coffee_blend_price_l4029_402946

/-- Given two blends of coffee, prove the price of the first blend -/
theorem coffee_blend_price 
  (price_blend2 : ℝ) 
  (total_weight : ℝ) 
  (total_price_per_pound : ℝ) 
  (weight_blend2 : ℝ) 
  (h1 : price_blend2 = 8) 
  (h2 : total_weight = 20) 
  (h3 : total_price_per_pound = 8.4) 
  (h4 : weight_blend2 = 12) : 
  ∃ (price_blend1 : ℝ), price_blend1 = 9 := by
sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l4029_402946


namespace NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l4029_402906

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l4029_402906


namespace NUMINAMATH_CALUDE_set_operations_l4029_402944

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 * x - 4 ≥ 0}

-- Define the theorem
theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 2 ∨ x ≥ 3}) ∧
  ((Set.univ \ A) ∩ (Set.univ \ B) = {x | x < -1}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l4029_402944


namespace NUMINAMATH_CALUDE_fraction_of_girls_at_joint_event_l4029_402949

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_in_school (s : School) : ℚ :=
  (s.total_students : ℚ) * s.girl_ratio / (s.boy_ratio + s.girl_ratio)

/-- Theorem stating that the fraction of girls at the joint event is 5/7 -/
theorem fraction_of_girls_at_joint_event 
  (school_a : School) 
  (school_b : School) 
  (ha : school_a = ⟨300, 3, 2⟩) 
  (hb : school_b = ⟨240, 3, 4⟩) : 
  (girls_in_school school_a + girls_in_school school_b) / 
  (school_a.total_students + school_b.total_students : ℚ) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_at_joint_event_l4029_402949


namespace NUMINAMATH_CALUDE_custom_op_difference_l4029_402931

/-- Custom operation @ defined as x@y = xy - 3x -/
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

/-- Theorem stating that (6@2)-(2@6) = -12 -/
theorem custom_op_difference : at_op 6 2 - at_op 2 6 = -12 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_difference_l4029_402931


namespace NUMINAMATH_CALUDE_min_cost_50_percent_alloy_l4029_402937

/-- Represents a gold alloy with its gold percentage and cost per ounce -/
structure GoldAlloy where
  percentage : Rat
  cost : Rat

/-- Theorem stating the minimum cost per ounce to create a 50% gold alloy -/
theorem min_cost_50_percent_alloy 
  (alloy40 : GoldAlloy) 
  (alloy60 : GoldAlloy)
  (alloy90 : GoldAlloy)
  (h1 : alloy40.percentage = 40/100)
  (h2 : alloy60.percentage = 60/100)
  (h3 : alloy90.percentage = 90/100)
  (h4 : alloy40.cost = 200)
  (h5 : alloy60.cost = 300)
  (h6 : alloy90.cost = 400) :
  ∃ (x y z : Rat),
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    (x * alloy40.percentage + y * alloy60.percentage + z * alloy90.percentage) / (x + y + z) = 1/2 ∧
    (x * alloy40.cost + y * alloy60.cost + z * alloy90.cost) / (x + y + z) = 240 ∧
    ∀ (a b c : Rat),
      a ≥ 0 → b ≥ 0 → c ≥ 0 →
      (a * alloy40.percentage + b * alloy60.percentage + c * alloy90.percentage) / (a + b + c) = 1/2 →
      (a * alloy40.cost + b * alloy60.cost + c * alloy90.cost) / (a + b + c) ≥ 240 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_50_percent_alloy_l4029_402937


namespace NUMINAMATH_CALUDE_base_three_20201_equals_181_l4029_402909

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_three_20201_equals_181 :
  base_three_to_ten [1, 0, 2, 0, 2] = 181 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20201_equals_181_l4029_402909


namespace NUMINAMATH_CALUDE_rectangle_width_equals_three_l4029_402924

theorem rectangle_width_equals_three (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 9)
  (h2 : rect_length = 27)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_three_l4029_402924


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4029_402986

theorem quadratic_equation_solution (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8) - 45 = 0) → b = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4029_402986


namespace NUMINAMATH_CALUDE_max_ratio_squared_l4029_402905

theorem max_ratio_squared (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_geq_b : a ≥ b)
  (h_eq : a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = Real.sqrt ((a - x)^2 + (b - y)^2))
  (h_bounds : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b)
  (h_right_triangle : (a - b + x)^2 + (b - a + y)^2 = a^2 + b^2) :
  (∀ ρ : ℝ, a ≤ ρ * b → ρ^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l4029_402905


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l4029_402967

theorem rectangle_area_problem (a b : ℝ) :
  (∀ (a b : ℝ), 
    ((a + 3) * b - a * b = 12) ∧
    ((a + 3) * (b + 3) - (a + 3) * b = 24)) →
  a * b = 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l4029_402967


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l4029_402961

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l4029_402961


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l4029_402942

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (81^2 - (m + 3) * 81 + m + 2 = 0) → m = 79 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l4029_402942


namespace NUMINAMATH_CALUDE_problem_solution_l4029_402959

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 5 + 1520 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4029_402959


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4029_402955

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 = 5 → b^2 = 12 → c^2 = a^2 + b^2 →
  c^2 = 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4029_402955


namespace NUMINAMATH_CALUDE_voyage_year_difference_l4029_402932

def zheng_he_voyage_year : ℕ := 2005 - 600
def columbus_voyage_year : ℕ := 1492

theorem voyage_year_difference : columbus_voyage_year - zheng_he_voyage_year = 87 := by
  sorry

end NUMINAMATH_CALUDE_voyage_year_difference_l4029_402932


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l4029_402993

/-- Given three lines that intersect at the same point, prove that k = -5 --/
theorem intersecting_lines_k_value (x y k : ℝ) :
  (y = 5 * x + 3) ∧
  (y = -2 * x - 25) ∧
  (y = 3 * x + k) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l4029_402993


namespace NUMINAMATH_CALUDE_fraction_equality_l4029_402903

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) (h1 : a / b = 1 / 3) :
  (2 * a + b) / (a - b) = -5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4029_402903


namespace NUMINAMATH_CALUDE_inequality_solution_l4029_402972

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + 
   (Real.sqrt (Real.sqrt (3 * x^2 + 6 * x + 7) + 
               Real.sqrt (5 * x^2 + 10 * x + 14) + 
               x^2 + 2 * x - 4))^(1/4) ≤ 18 - 8 * x) ↔ 
  x = -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4029_402972


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4029_402969

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4029_402969
